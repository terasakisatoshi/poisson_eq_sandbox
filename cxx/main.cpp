// Build: ./build.sh  (g++ -O3 -std=c++23 -o poisson main.cpp)
//
// ------------------------------------------------------------
// Problem definition
//
//   -Δu = f    in Ω = (0,1) × (0,1)
//      u = 0   on ∂Ω
//
// Exact solution:
//
//   u(x,y) = sin(πx) sin(πy)
//
// Therefore
//
//   f(x,y) = 2π² sin(πx) sin(πy)
//
// Same Jacobi stencil as julia/poisson.jl. Grids are an owning value type
// indexed with `operator()(i, j)` in column-major order, and successive
// iterates are exchanged with an O(1) swap. Only interior points are
// written, so both buffers stay zero on the Dirichlet boundary. The
// update-width reduction and the progress line are confined to every
// 1000th sweep (as in rust_unsafe), which keeps the hot sweeps free of the
// output call so the inner loop vectorizes. The figure is a
// self-contained PNG encoder (stored deflate blocks), mirroring the
// Fortran version.
// ------------------------------------------------------------

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <print>
#include <span>
#include <string_view>
#include <vector>

namespace {

using Byte = std::uint8_t;

namespace config {
constexpr std::size_t grid_size = 401;
constexpr double tolerance = 1e-10;
constexpr std::size_t max_iterations = 100'000;
constexpr std::size_t report_interval = 1'000;
}  // namespace config

double u_exact(double x, double y) {
    return std::sin(std::numbers::pi * x) * std::sin(std::numbers::pi * y);
}

double f_rhs(double x, double y) {
    constexpr double two_pi_squared = 2.0 * std::numbers::pi * std::numbers::pi;
    return two_pi_squared * std::sin(std::numbers::pi * x) * std::sin(std::numbers::pi * y);
}

// Owning dense grid of scalar samples, column-major so that the first
// index (i) is contiguous.
class Grid {
public:
    explicit Grid(std::size_t size) : size_(size), values_(size * size, 0.0) {}

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    [[nodiscard]] std::span<const double> data() const noexcept { return values_; }
    [[nodiscard]] std::span<double> data() noexcept { return values_; }

    double& operator()(std::size_t i, std::size_t j) noexcept {
        return values_[i + size_ * j];
    }

    double operator()(std::size_t i, std::size_t j) const noexcept {
        return values_[i + size_ * j];
    }

    void swap(Grid& other) noexcept {
        std::swap(size_, other.size_);
        values_.swap(other.values_);
    }

private:
    std::size_t size_;
    std::vector<double> values_;
};

struct JacobiReport {
    std::size_t iterations;
    double update_error;
};

// Update-width Jacobi iteration for the five-point Laplacian. The result
// is always the buffer that holds the latest iterate.
class JacobiSolver {
public:
    explicit JacobiSolver(double h) : h_squared_(h * h) {}

    [[nodiscard]] JacobiReport solve(Grid& u, Grid& u_next, const Grid& rhs) const {
        double error = std::numeric_limits<double>::infinity();
        std::size_t iterations = 0;

        for (std::size_t iteration = 1; iteration <= config::max_iterations; ++iteration) {
            // Track the update width (and print) only at reported iterations.
            // Keeping the output call out of most sweeps lets the compiler
            // vectorize the inner loop; see `sweep`.
            const bool report = iteration % config::report_interval == 0 ||
                                iteration == config::max_iterations;
            if (report) {
                error = sweep<true>(u, u_next, rhs);
            } else {
                sweep<false>(u, u_next, rhs);
            }
            u.swap(u_next);
            iterations = iteration;

            if (report) {
                std::println("iteration = {:6}, update error = {:.6e}", iteration, error);
                if (error < config::tolerance) {
                    break;
                }
            }
        }

        return {iterations, error};
    }

private:
    // One Jacobi sweep over the interior. When `Track` is false the max
    // reduction is compiled out, so there is no `println` in the outer loop
    // and the inner loop vectorizes.
    template <bool Track>
    double sweep(Grid& u, Grid& u_next, const Grid& rhs) const {
        const std::size_t n = u.size();
        double error = 0.0;

        for (std::size_t j = 1; j + 1 < n; ++j) {
            for (std::size_t i = 1; i + 1 < n; ++i) {
                const double value = 0.25 * (u(i + 1, j) + u(i - 1, j) + u(i, j + 1) +
                                             u(i, j - 1) + h_squared_ * rhs(i, j));
                if constexpr (Track) {
                    error = std::max(error, std::abs(value - u(i, j)));
                }
                u_next(i, j) = value;
            }
        }

        return error;
    }

    double h_squared_;
};

// ------------------------------------------------------------
// Minimal dependency-free PNG output (8-bit RGB, stored deflate).
// ------------------------------------------------------------

std::uint32_t crc32(std::span<const Byte> data) {
    std::uint32_t crc = 0xFFFFFFFFu;
    for (const Byte byte : data) {
        crc ^= byte;
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

std::uint32_t adler32(std::span<const Byte> data) {
    std::uint32_t s1 = 1;
    std::uint32_t s2 = 0;
    for (const Byte byte : data) {
        s1 = (s1 + byte) % 65521u;
        s2 = (s2 + s1) % 65521u;
    }
    return (s2 << 16) | s1;
}

void put_u32be(std::vector<Byte>& out, std::uint32_t value) {
    out.push_back(static_cast<Byte>(value >> 24));
    out.push_back(static_cast<Byte>(value >> 16));
    out.push_back(static_cast<Byte>(value >> 8));
    out.push_back(static_cast<Byte>(value));
}

void append_chunk(std::vector<Byte>& png, std::string_view type,
                  std::span<const Byte> payload) {
    put_u32be(png, static_cast<std::uint32_t>(payload.size()));
    const std::size_t crc_start = png.size();
    png.insert(png.end(), type.begin(), type.end());
    png.insert(png.end(), payload.begin(), payload.end());
    put_u32be(png, crc32(std::span{png}.subspan(crc_start)));
}

struct Rgb {
    Byte r;
    Byte g;
    Byte b;
};

class Canvas {
public:
    Canvas(int width, int height)
        : width_(width), height_(height), pixels_(std::size_t(width) * height * 3, 0xFF) {}

    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }

    void set(int x, int y, Rgb color) noexcept {
        if (x < 0 || x >= width_ || y < 0 || y >= height_) {
            return;
        }
        const std::size_t k = (static_cast<std::size_t>(y) * width_ + x) * 3;
        pixels_[k] = color.r;
        pixels_[k + 1] = color.g;
        pixels_[k + 2] = color.b;
    }

    void write_png(const std::filesystem::path& path) const;

private:
    int width_;
    int height_;
    std::vector<Byte> pixels_;
};

void Canvas::write_png(const std::filesystem::path& path) const {
    // Raw scanlines with filter byte 0 per row.
    std::vector<Byte> raw;
    raw.reserve(static_cast<std::size_t>(height_) * (1 + 3 * width_));
    for (int y = 0; y < height_; ++y) {
        raw.push_back(0);
        const std::size_t offset = static_cast<std::size_t>(y) * width_ * 3;
        raw.insert(raw.end(), pixels_.begin() + offset,
                   pixels_.begin() + offset + static_cast<std::size_t>(width_) * 3);
    }

    // zlib stream with stored (uncompressed) deflate blocks.
    std::vector<Byte> deflate;
    deflate.push_back(0x78);
    deflate.push_back(0x01);
    const auto emit = [&deflate](std::span<const Byte> block, bool final_block) {
        const std::uint16_t length = static_cast<std::uint16_t>(block.size());
        const std::uint16_t inverted = static_cast<std::uint16_t>(~length);
        deflate.push_back(final_block ? 0x01 : 0x00);
        deflate.push_back(static_cast<Byte>(length & 0xFF));
        deflate.push_back(static_cast<Byte>(length >> 8));
        deflate.push_back(static_cast<Byte>(inverted & 0xFF));
        deflate.push_back(static_cast<Byte>(inverted >> 8));
        deflate.insert(deflate.end(), block.begin(), block.end());
    };
    for (std::size_t position = 0; position < raw.size(); position += 65535) {
        const std::size_t take = std::min<std::size_t>(65535, raw.size() - position);
        emit(std::span{raw}.subspan(position, take), position + take == raw.size());
    }
    put_u32be(deflate, adler32(raw));

    std::vector<Byte> header;
    put_u32be(header, static_cast<std::uint32_t>(width_));
    put_u32be(header, static_cast<std::uint32_t>(height_));
    header.push_back(8);  // bit depth
    header.push_back(2);  // color type: truecolor
    header.push_back(0);  // compression
    header.push_back(0);  // filter
    header.push_back(0);  // interlace

    std::vector<Byte> png = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    append_chunk(png, "IHDR", header);
    append_chunk(png, "IDAT", deflate);
    append_chunk(png, "IEND", {});

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::println("failed to write PNG");
        return;
    }
    out.write(reinterpret_cast<const char*>(png.data()),
              static_cast<std::streamsize>(png.size()));
}

// ------------------------------------------------------------
// Figure drawing (three heatmap panels), mirroring the Fortran version.
// ------------------------------------------------------------

Rgb height_color(double z) {
    const double t = std::clamp(z, 0.0, 1.0);
    constexpr double saturation = 0.70;
    const double hue = 0.65 * (1.0 - t) * 6.0;
    const double x = saturation * (1.0 - std::abs(std::fmod(hue, 2.0) - 1.0));

    double r = 0.0;
    double g = 0.0;
    double b = 0.0;
    if (hue < 1.0) {
        r = saturation, g = x;
    } else if (hue < 2.0) {
        r = x, g = saturation;
    } else if (hue < 3.0) {
        g = saturation, b = x;
    } else if (hue < 4.0) {
        g = x, b = saturation;
    } else if (hue < 5.0) {
        r = x, b = saturation;
    } else {
        r = saturation, b = x;
    }

    constexpr double offset = 0.45 - 0.5 * saturation;
    const auto to_byte = [](double v) {
        return static_cast<Byte>(std::lround(255.0 * v));
    };
    return {to_byte(r + offset), to_byte(g + offset), to_byte(b + offset)};
}

Rgb viridis(double t) {
    struct Stop {
        double position;
        Rgb color;
    };
    constexpr std::array stops = {
        Stop{0.00, {68, 1, 84}},   Stop{0.25, {59, 82, 139}},
        Stop{0.50, {33, 145, 140}}, Stop{0.75, {94, 201, 98}},
        Stop{1.00, {253, 231, 37}},
    };
    t = std::clamp(t, 0.0, 1.0);

    std::size_t k = 0;
    while (k + 2 < stops.size() && t > stops[k + 1].position) {
        ++k;
    }
    const double span = stops[k + 1].position - stops[k].position;
    const double a = std::abs(span) < 1e-12 ? 0.0 : (t - stops[k].position) / span;
    const auto mix = [a](Byte from, Byte to) {
        return static_cast<Byte>(std::lround(from + a * (static_cast<double>(to) - from)));
    };
    return {mix(stops[k].color.r, stops[k + 1].color.r),
            mix(stops[k].color.g, stops[k + 1].color.g),
            mix(stops[k].color.b, stops[k + 1].color.b)};
}

struct Glyph {
    char character;
    std::array<Byte, 7> rows;
};

constexpr std::array glyphs = {
    Glyph{'A', {14, 17, 17, 31, 17, 17, 17}}, Glyph{'E', {31, 1, 1, 15, 1, 1, 31}},
    Glyph{'I', {31, 4, 4, 4, 4, 4, 31}},      Glyph{'L', {1, 1, 1, 1, 1, 1, 31}},
    Glyph{'N', {17, 19, 21, 21, 25, 17, 17}}, Glyph{'O', {14, 17, 17, 17, 17, 17, 14}},
    Glyph{'R', {15, 17, 17, 15, 5, 9, 17}},   Glyph{'S', {14, 17, 1, 14, 16, 17, 14}},
    Glyph{'T', {31, 4, 4, 4, 4, 4, 4}},       Glyph{'U', {17, 17, 17, 17, 17, 17, 14}},
    Glyph{'X', {17, 17, 10, 4, 10, 17, 17}},  Glyph{'a', {0, 0, 14, 16, 30, 17, 30}},
    Glyph{'b', {1, 1, 15, 17, 17, 17, 15}},   Glyph{'c', {0, 0, 14, 17, 1, 17, 14}},
    Glyph{'e', {0, 0, 14, 17, 31, 1, 14}},    Glyph{'i', {4, 0, 4, 4, 4, 4, 14}},
    Glyph{'l', {6, 4, 4, 4, 4, 4, 12}},       Glyph{'m', {0, 0, 11, 21, 21, 21, 21}},
    Glyph{'n', {0, 0, 15, 17, 17, 17, 17}},   Glyph{'o', {0, 0, 14, 17, 17, 17, 14}},
    Glyph{'r', {0, 0, 13, 19, 1, 1, 1}},      Glyph{'s', {0, 0, 30, 1, 14, 16, 15}},
    Glyph{'t', {4, 4, 15, 4, 4, 4, 12}},      Glyph{'u', {0, 0, 17, 17, 17, 17, 14}},
    Glyph{'v', {0, 0, 17, 17, 17, 10, 4}},    Glyph{'x', {0, 0, 17, 10, 4, 10, 17}},
};

std::array<Byte, 7> glyph_for(char character) {
    const auto found = std::ranges::find(glyphs, character, &Glyph::character);
    return found == glyphs.end() ? std::array<Byte, 7>{} : found->rows;
}

void draw_title(Canvas& canvas, int left, int top, std::string_view title) {
    constexpr Rgb black{0, 0, 0};
    int cursor = left;
    for (const char character : title) {
        const auto rows = glyph_for(character);
        for (int row = 0; row < 7; ++row) {
            for (int column = 0; column < 5; ++column) {
                if (((rows[row] >> column) & 1) == 0) {
                    continue;
                }
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        canvas.set(cursor + 2 * column + dx, top + 2 * row + dy, black);
                    }
                }
            }
        }
        cursor += 12;
    }
}

void draw_frame(Canvas& canvas, int left, int top, int right, int bottom) {
    constexpr Rgb black{0, 0, 0};
    for (int x = left; x <= right; ++x) {
        canvas.set(x, top, black);
        canvas.set(x, bottom, black);
    }
    for (int y = top; y <= bottom; ++y) {
        canvas.set(left, y, black);
        canvas.set(right, y, black);
    }
}

void draw_heatmap(Canvas& canvas, int left, int top, int right, int bottom,
                  std::span<const double> field, std::size_t n, double zmin,
                  double zmax, bool use_viridis) {
    const double range = zmax > zmin ? zmax - zmin : 1.0;
    const int width = std::max(right - left, 1);
    const int height = std::max(bottom - top, 1);

    for (int py = top; py <= bottom; ++py) {
        const double y = 1.0 - static_cast<double>(py - top) / height;
        const auto j = static_cast<std::size_t>(std::clamp(
            std::lround(y * static_cast<double>(n - 1)), 0L, static_cast<long>(n - 1)));
        for (int px = left; px <= right; ++px) {
            const double x = static_cast<double>(px - left) / width;
            const auto i = static_cast<std::size_t>(std::clamp(
                std::lround(x * static_cast<double>(n - 1)), 0L, static_cast<long>(n - 1)));
            const double t = std::clamp((field[i + n * j] - zmin) / range, 0.0, 1.0);
            canvas.set(px, py, use_viridis ? viridis(t) : height_color(t));
        }
    }
}

void save_plot(const std::filesystem::path& path, const Grid& u, const Grid& ue,
               const Grid& error, double max_error) {
    constexpr int width = 1500;
    constexpr int height = 450;
    constexpr int padding = 16;
    const int top = padding + 24;
    const int bottom = height - padding;

    struct Panel {
        std::string_view title;
        const Grid* field;
        double zmax;
        bool use_viridis;
    };
    const std::array panels = {
        Panel{"Exact solution", &ue, 1.0, false},
        Panel{"Numerical solution", &u, 1.0, false},
        Panel{"Absolute error", &error, std::max(max_error, 1e-30), true},
    };

    Canvas canvas(width, height);
    for (std::size_t p = 0; p < panels.size(); ++p) {
        const int left = static_cast<int>(p) * (width / 3) + padding;
        const int right = static_cast<int>(p + 1) * (width / 3) - padding;
        draw_heatmap(canvas, left, top, right, bottom, panels[p].field->data(),
                     panels[p].field->size(), 0.0, panels[p].zmax,
                     panels[p].use_viridis);
        draw_title(canvas, left, padding, panels[p].title);
        draw_frame(canvas, left, top, right, bottom);
    }
    canvas.write_png(path);
}

}  // namespace

int main() {
    constexpr std::size_t n = config::grid_size;
    const double h = 1.0 / static_cast<double>(n - 1);

    std::println("N = {}\nh = {:.6e}", n, h);

    Grid rhs(n);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            rhs(i, j) = f_rhs(static_cast<double>(i) * h, static_cast<double>(j) * h);
        }
    }

    Grid u(n);
    Grid u_next(n);

    const auto started = std::chrono::steady_clock::now();
    const JacobiReport report = JacobiSolver{h}.solve(u, u_next, rhs);
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - started;

    std::println(
        "\ntime = {:.6f} seconds\nJacobi iterations = {}\nfinal update error = {:.6e}",
        elapsed.count(), report.iterations, report.update_error);

    Grid ue(n);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            ue(i, j) = u_exact(static_cast<double>(i) * h, static_cast<double>(j) * h);
        }
    }

    Grid error(n);
    double max_error = 0.0;
    double sum_squared = 0.0;
    for (std::size_t k = 0; k < u.data().size(); ++k) {
        const double difference = u.data()[k] - ue.data()[k];
        error.data()[k] = std::abs(difference);
        max_error = std::max(max_error, error.data()[k]);
        sum_squared += difference * difference;
    }
    const double l2_error = std::sqrt(sum_squared * h * h);

    std::println("\nmax error = {:.6e}\nL2 error  = {:.6e}", max_error,
                 l2_error / std::sqrt(static_cast<double>(n)));

    const std::filesystem::path output = "poisson_jacobi.png";
    save_plot(output, u, ue, error, max_error);
    std::println("saved {}", output.string());

    return 0;
}
