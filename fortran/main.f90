! Build: ./build.sh  (gfortran -O3 -o poisson main.f90)
!
! ------------------------------------------------------------
! Problem definition
!
!   -Δu = f    in Ω = (0,1) × (0,1)
!      u = 0   on ∂Ω
!
! Exact solution:
!
!   u(x,y) = sin(πx) sin(πy)
!
! Therefore
!
!   f(x,y) = 2π² sin(πx) sin(πy)
!
! Same Jacobi stencil as julia_unsafe/poisson.jl. gfortran does not
! insert bounds checks unless -fcheck=bounds is given, so -O3 is
! already the @inbounds equivalent. -ffast-math is @fastmath.
! Buffers are pointer-swapped instead of copied; both stay zero on
! the Dirichlet boundary because only interior points are written.
! ------------------------------------------------------------

program poisson
  use, intrinsic :: iso_fortran_env, only: dp => real64, int8, int32, int64
  implicit none

  integer, parameter :: n = 401
  real(dp), parameter :: tol = 1.0e-10_dp
  integer, parameter :: maxiter = 100000
  real(dp), parameter :: pi = 4.0_dp * atan(1.0_dp)

  real(dp) :: h, duration, update_error, max_error, l2_error
  real(dp), allocatable, target :: x(:), y(:), ua(:,:), ub(:,:), rhs(:,:), ue(:,:), err(:,:)
  real(dp), pointer, contiguous :: u(:,:), u_new(:,:)
  integer :: i, j, iterations
  integer(int64) :: t0, t1, clock_rate

  allocate (x(n), y(n), ua(n, n), ub(n, n), rhs(n, n), ue(n, n), err(n, n))

  h = 1.0_dp / real(n - 1, dp)
  do i = 1, n
    x(i) = real(i - 1, dp) * h
    y(i) = x(i)
  end do

  write (*, '(a, i0)') 'N = ', n
  write (*, '(a, es15.6)') 'h = ', h

  ua = 0.0_dp
  ub = 0.0_dp
  u => ua
  u_new => ub
  do j = 1, n
    do i = 1, n
      rhs(i, j) = f_rhs(x(i), y(j))
    end do
  end do

  call system_clock(t0, clock_rate)
  call jacobi(u, u_new, rhs, h, tol, maxiter, iterations, update_error)
  call system_clock(t1)

  duration = real(t1 - t0, dp) / real(clock_rate, dp)

  write (*, '()')
  write (*, '(a, f0.6, a)') 'time = ', duration, ' seconds'
  write (*, '(a, i0)') 'Jacobi iterations = ', iterations
  write (*, '(a, es15.6)') 'final update error = ', update_error

  do j = 1, n
    do i = 1, n
      ue(i, j) = u_exact(x(i), y(j))
    end do
  end do

  err = abs(u - ue)
  max_error = maxval(err)
  l2_error = sqrt(sum((u - ue)**2) * h**2)

  write (*, '()')
  write (*, '(a, es15.6)') 'max error = ', max_error
  write (*, '(a, es15.6)') 'L2 error  = ', l2_error / sqrt(real(n, dp))

  call save_plot('poisson_jacobi.png', u, ue, err, n, max_error)
  write (*, '(a)') 'saved poisson_jacobi.png'

contains

  pure function u_exact(xx, yy) result(val)
    real(dp), intent(in) :: xx, yy
    real(dp) :: val
    val = sin(pi * xx) * sin(pi * yy)
  end function u_exact

  pure function f_rhs(xx, yy) result(val)
    real(dp), intent(in) :: xx, yy
    real(dp) :: val
    val = 2.0_dp * pi**2 * sin(pi * xx) * sin(pi * yy)
  end function f_rhs

  subroutine jacobi(u, u_new, rhs, h, tol, maxiter, iterations, update_error)
    real(dp), pointer, contiguous, intent(inout) :: u(:, :), u_new(:, :)
    real(dp), intent(in), contiguous :: rhs(:, :)
    real(dp), intent(in) :: h, tol
    integer, intent(in) :: maxiter
    integer, intent(out) :: iterations
    real(dp), intent(out) :: update_error

    integer :: nloc, iter, i, j
    real(dp) :: h2, val
    real(dp), pointer, contiguous :: tmp(:, :)

    nloc = size(u, 1)
    h2 = h**2
    update_error = huge(0.0_dp)
    iterations = 0

    do iter = 1, maxiter
      update_error = 0.0_dp

      ! Interior points only.
      ! Boundary points remain zero (Dirichlet).
      do j = 2, nloc - 1
        do i = 2, nloc - 1
          val = 0.25_dp * (u(i + 1, j) + u(i - 1, j) + u(i, j + 1) + u(i, j - 1) + &
                           h2 * rhs(i, j))
          update_error = max(update_error, abs(val - u(i, j)))
          u_new(i, j) = val
        end do
      end do

      tmp => u
      u => u_new
      u_new => tmp
      iterations = iter

      if (mod(iter, 1000) == 0) then
        write (*, '(a, i6, a, es15.6)') 'iteration = ', iter, ', update error = ', update_error
      end if

      if (update_error < tol) exit
    end do
  end subroutine jacobi

  subroutine save_plot(path, u, ue, err, n, max_error)
    character(len=*), intent(in) :: path
    real(dp), intent(in) :: u(:, :), ue(:, :), err(:, :), max_error
    integer, intent(in) :: n

    integer, parameter :: width = 1500, height = 450
    integer, allocatable :: img(:, :, :)
    integer :: p, x0, y0, x1, y1, pad

    allocate (img(width, height, 3))
    img = 255

    pad = 16
    y0 = pad + 24
    y1 = height - pad

    do p = 0, 2
      x0 = p * (width / 3) + pad
      x1 = (p + 1) * (width / 3) - pad
      select case (p)
      case (0)
        call draw_title(img, width, height, x0, pad, 'Exact solution')
        call draw_heatmap(img, width, height, x0, y0, x1, y1, ue, n, 0.0_dp, 1.0_dp, .false.)
      case (1)
        call draw_title(img, width, height, x0, pad, 'Numerical solution')
        call draw_heatmap(img, width, height, x0, y0, x1, y1, u, n, 0.0_dp, 1.0_dp, .false.)
      case (2)
        call draw_title(img, width, height, x0, pad, 'Absolute error')
        call draw_heatmap(img, width, height, x0, y0, x1, y1, err, n, 0.0_dp, &
                          max(max_error, 1.0e-30_dp), .true.)
      end select
      call draw_rect(img, width, height, x0, y0, x1, y1, 0, 0, 0)
    end do

    call write_png(path, img, width, height)
  end subroutine save_plot

  subroutine draw_heatmap(img, w, h, x0, y0, x1, y1, field, n, zmin, zmax, use_viridis)
    integer, intent(inout) :: img(:, :, :)
    integer, intent(in) :: w, h, x0, y0, x1, y1, n
    real(dp), intent(in) :: field(:, :), zmin, zmax
    logical, intent(in) :: use_viridis

    integer :: px, py, i, j, r, g, b
    real(dp) :: xx, yy, t, z, denom

    denom = zmax - zmin
    if (denom <= 0.0_dp) denom = 1.0_dp

    do py = y0, y1
      yy = 1.0_dp - real(py - y0, dp) / real(max(y1 - y0, 1), dp)
      j = min(n, max(1, nint(yy * real(n - 1, dp)) + 1))
      do px = x0, x1
        xx = real(px - x0, dp) / real(max(x1 - x0, 1), dp)
        i = min(n, max(1, nint(xx * real(n - 1, dp)) + 1))
        z = field(i, j)
        t = min(1.0_dp, max(0.0_dp, (z - zmin) / denom))
        if (use_viridis) then
          call viridis(t, r, g, b)
        else
          call height_color(t, r, g, b)
        end if
        if (px >= 1 .and. px <= w .and. py >= 1 .and. py <= h) then
          img(px, py, 1) = r
          img(px, py, 2) = g
          img(px, py, 3) = b
        end if
      end do
    end do
  end subroutine draw_heatmap

  subroutine height_color(z, r, g, b)
    real(dp), intent(in) :: z
    integer, intent(out) :: r, g, b
    real(dp) :: hh, s, l, c, x, m, rp, gp, bp, h6

    hh = 0.65_dp * (1.0_dp - min(1.0_dp, max(0.0_dp, z)))
    s = 0.70_dp
    l = 0.45_dp
    c = (1.0_dp - abs(2.0_dp * l - 1.0_dp)) * s
    h6 = hh * 6.0_dp
    x = c * (1.0_dp - abs(mod(h6, 2.0_dp) - 1.0_dp))
    if (h6 < 1.0_dp) then
      rp = c; gp = x; bp = 0.0_dp
    else if (h6 < 2.0_dp) then
      rp = x; gp = c; bp = 0.0_dp
    else if (h6 < 3.0_dp) then
      rp = 0.0_dp; gp = c; bp = x
    else if (h6 < 4.0_dp) then
      rp = 0.0_dp; gp = x; bp = c
    else if (h6 < 5.0_dp) then
      rp = x; gp = 0.0_dp; bp = c
    else
      rp = c; gp = 0.0_dp; bp = x
    end if
    m = l - 0.5_dp * c
    r = nint(255.0_dp * (rp + m))
    g = nint(255.0_dp * (gp + m))
    b = nint(255.0_dp * (bp + m))
  end subroutine height_color

  subroutine viridis(t, r, g, b)
    real(dp), intent(in) :: t
    integer, intent(out) :: r, g, b

    real(dp), parameter :: stops(3, 5) = reshape([ &
      68.0_dp, 1.0_dp, 84.0_dp, &
      59.0_dp, 82.0_dp, 139.0_dp, &
      33.0_dp, 145.0_dp, 140.0_dp, &
      94.0_dp, 201.0_dp, 98.0_dp, &
      253.0_dp, 231.0_dp, 37.0_dp], [3, 5])
    real(dp), parameter :: ts(5) = [0.0_dp, 0.25_dp, 0.50_dp, 0.75_dp, 1.0_dp]
    real(dp) :: tt, a
    integer :: k

    tt = min(1.0_dp, max(0.0_dp, t))
    k = 1
    do while (k < 4 .and. tt > ts(k + 1))
      k = k + 1
    end do
    if (abs(ts(k + 1) - ts(k)) < 1.0e-12_dp) then
      a = 0.0_dp
    else
      a = (tt - ts(k)) / (ts(k + 1) - ts(k))
    end if
    r = nint(stops(1, k) + a * (stops(1, k + 1) - stops(1, k)))
    g = nint(stops(2, k) + a * (stops(2, k + 1) - stops(2, k)))
    b = nint(stops(3, k) + a * (stops(3, k + 1) - stops(3, k)))
  end subroutine viridis

  subroutine draw_rect(img, w, h, x0, y0, x1, y1, r, g, b)
    integer, intent(inout) :: img(:, :, :)
    integer, intent(in) :: w, h, x0, y0, x1, y1, r, g, b
    integer :: px, py

    do px = max(1, x0), min(w, x1)
      if (y0 >= 1 .and. y0 <= h) then
        img(px, y0, 1) = r; img(px, y0, 2) = g; img(px, y0, 3) = b
      end if
      if (y1 >= 1 .and. y1 <= h) then
        img(px, y1, 1) = r; img(px, y1, 2) = g; img(px, y1, 3) = b
      end if
    end do
    do py = max(1, y0), min(h, y1)
      if (x0 >= 1 .and. x0 <= w) then
        img(x0, py, 1) = r; img(x0, py, 2) = g; img(x0, py, 3) = b
      end if
      if (x1 >= 1 .and. x1 <= w) then
        img(x1, py, 1) = r; img(x1, py, 2) = g; img(x1, py, 3) = b
      end if
    end do
  end subroutine draw_rect

  subroutine draw_title(img, w, h, x0, y0, title)
    integer, intent(inout) :: img(:, :, :)
    integer, intent(in) :: w, h, x0, y0
    character(len=*), intent(in) :: title
    integer :: k, cx, cy, gi, row, col, bit, px, py
    integer(int8) :: glyph(7)

    cx = x0
    cy = y0
    do k = 1, len_trim(title)
      call font5x7(title(k:k), glyph)
      do row = 0, 6
        do col = 0, 4
          bit = iand(ishft(int(glyph(row + 1)), -col), 1)
          if (bit == 0) cycle
          do py = 0, 1
            do px = 0, 1
              gi = cx + 2 * col + px
              if (gi >= 1 .and. gi <= w .and. cy + 2 * row + py >= 1 .and. &
                  cy + 2 * row + py <= h) then
                img(gi, cy + 2 * row + py, :) = 0
              end if
            end do
          end do
        end do
      end do
      cx = cx + 12
    end do
  end subroutine draw_title

  subroutine font5x7(ch, glyph)
    character(len=1), intent(in) :: ch
    integer(int8), intent(out) :: glyph(7)

    glyph = 0_int8
    select case (ch)
    case ('A'); glyph = int([14, 17, 17, 31, 17, 17, 17], int8)
    case ('E'); glyph = int([31, 1, 1, 15, 1, 1, 31], int8)
    case ('I'); glyph = int([31, 4, 4, 4, 4, 4, 31], int8)
    case ('L'); glyph = int([1, 1, 1, 1, 1, 1, 31], int8)
    case ('N'); glyph = int([17, 19, 21, 21, 25, 17, 17], int8)
    case ('O'); glyph = int([14, 17, 17, 17, 17, 17, 14], int8)
    case ('R'); glyph = int([15, 17, 17, 15, 5, 9, 17], int8)
    case ('S'); glyph = int([14, 17, 1, 14, 16, 17, 14], int8)
    case ('T'); glyph = int([31, 4, 4, 4, 4, 4, 4], int8)
    case ('U'); glyph = int([17, 17, 17, 17, 17, 17, 14], int8)
    case ('X'); glyph = int([17, 17, 10, 4, 10, 17, 17], int8)
    case ('a'); glyph = int([0, 0, 14, 16, 30, 17, 30], int8)
    case ('b'); glyph = int([1, 1, 15, 17, 17, 17, 15], int8)
    case ('c'); glyph = int([0, 0, 14, 17, 1, 17, 14], int8)
    case ('e'); glyph = int([0, 0, 14, 17, 31, 1, 14], int8)
    case ('i'); glyph = int([4, 0, 4, 4, 4, 4, 14], int8)
    case ('l'); glyph = int([6, 4, 4, 4, 4, 4, 12], int8)
    case ('m'); glyph = int([0, 0, 11, 21, 21, 21, 21], int8)
    case ('n'); glyph = int([0, 0, 15, 17, 17, 17, 17], int8)
    case ('o'); glyph = int([0, 0, 14, 17, 17, 17, 14], int8)
    case ('r'); glyph = int([0, 0, 13, 19, 1, 1, 1], int8)
    case ('s'); glyph = int([0, 0, 30, 1, 14, 16, 15], int8)
    case ('t'); glyph = int([4, 4, 15, 4, 4, 4, 12], int8)
    case ('u'); glyph = int([0, 0, 17, 17, 17, 17, 14], int8)
    case ('v'); glyph = int([0, 0, 17, 17, 17, 10, 4], int8)
    case (' '); glyph = 0_int8
    case default; glyph = 0_int8
    end select
  end subroutine font5x7

  subroutine write_png(path, img, width, height)
    character(len=*), intent(in) :: path
    integer, intent(in) :: img(:, :, :), width, height

    integer :: raw_len, nblocks, i, y, x, pos, blk, remain, take, unit, iostat
    integer(int8), allocatable :: raw(:), idat(:), filebuf(:)
    integer(int32) :: adler, crc
    integer :: file_len

    raw_len = height * (1 + 3 * width)
    allocate (raw(raw_len))
    pos = 1
    do y = 1, height
      raw(pos) = 0_int8
      pos = pos + 1
      do x = 1, width
        raw(pos) = to_u8(img(x, y, 1))
        raw(pos + 1) = to_u8(img(x, y, 2))
        raw(pos + 2) = to_u8(img(x, y, 3))
        pos = pos + 3
      end do
    end do

    nblocks = (raw_len + 65534) / 65535
    allocate (idat(2 + raw_len + 5 * nblocks + 4))
    idat(1) = to_u8(int(z'78'))
    idat(2) = to_u8(1)
    pos = 3
    remain = raw_len
    i = 1
    do blk = 1, nblocks
      take = min(65535, remain)
      if (blk == nblocks) then
        idat(pos) = to_u8(1)
      else
        idat(pos) = to_u8(0)
      end if
      idat(pos + 1) = to_u8(iand(take, 255))
      idat(pos + 2) = to_u8(iand(ishft(take, -8), 255))
      idat(pos + 3) = to_u8(iand(ieor(take, int(z'FFFF')), 255))
      idat(pos + 4) = to_u8(iand(ishft(ieor(take, int(z'FFFF')), -8), 255))
      idat(pos + 5:pos + 4 + take) = raw(i:i + take - 1)
      pos = pos + 5 + take
      i = i + take
      remain = remain - take
    end do

    adler = adler32(raw, raw_len)
    call put_u32be(idat, pos, adler)
    pos = pos + 4

    ! signature + IHDR(4+4+13+4) + IDAT(4+4+pos-1+4) + IEND(4+4+4)
    file_len = 8 + 25 + (12 + pos - 1) + 12
    allocate (filebuf(file_len))
    filebuf(1:8) = [to_u8(137), to_u8(80), to_u8(78), to_u8(71), &
                    to_u8(13), to_u8(10), to_u8(26), to_u8(10)]

    i = 9
    call put_u32be(filebuf, i, 13)
    filebuf(i + 4:i + 7) = int([iachar('I'), iachar('H'), iachar('D'), iachar('R')], int8)
    call put_u32be(filebuf, i + 8, width)
    call put_u32be(filebuf, i + 12, height)
    filebuf(i + 16) = 8_int8
    filebuf(i + 17) = 2_int8
    filebuf(i + 18:i + 20) = 0_int8
    crc = crc32(filebuf(i + 4:i + 20), 17)
    call put_u32be(filebuf, i + 21, crc)

    i = i + 25
    call put_u32be(filebuf, i, pos - 1)
    filebuf(i + 4:i + 7) = int([iachar('I'), iachar('D'), iachar('A'), iachar('T')], int8)
    filebuf(i + 8:i + 7 + pos - 1) = idat(1:pos - 1)
    crc = crc32(filebuf(i + 4:i + 7 + pos - 1), 4 + pos - 1)
    call put_u32be(filebuf, i + 8 + pos - 1, crc)

    i = i + 12 + pos - 1
    call put_u32be(filebuf, i, 0)
    filebuf(i + 4:i + 7) = int([iachar('I'), iachar('E'), iachar('N'), iachar('D')], int8)
    crc = crc32(filebuf(i + 4:i + 7), 4)
    call put_u32be(filebuf, i + 8, crc)

    open (newunit=unit, file=path, access='stream', form='unformatted', &
          status='replace', iostat=iostat)
    if (iostat /= 0) then
      write (*, '(a)') 'failed to write PNG'
      return
    end if
    write (unit) filebuf
    close (unit)
  end subroutine write_png

  pure function to_u8(v) result(b)
    integer, intent(in) :: v
    integer(int8) :: b
    integer :: u
    u = iand(v, 255)
    if (u > 127) then
      b = int(u - 256, int8)
    else
      b = int(u, int8)
    end if
  end function to_u8

  subroutine put_u32be(buf, pos, val)
    integer(int8), intent(inout) :: buf(:)
    integer, intent(in) :: pos
    integer(int32), intent(in) :: val
    integer(int32) :: v

    v = val
    buf(pos) = to_u8(int(iand(ishft(v, -24), 255)))
    buf(pos + 1) = to_u8(int(iand(ishft(v, -16), 255)))
    buf(pos + 2) = to_u8(int(iand(ishft(v, -8), 255)))
    buf(pos + 3) = to_u8(int(iand(v, 255)))
  end subroutine put_u32be

  integer(int32) function adler32(data, n) result(out)
    integer(int8), intent(in) :: data(:)
    integer, intent(in) :: n
    integer(int32) :: s1, s2, i, b

    s1 = 1
    s2 = 0
    do i = 1, n
      b = iand(int(data(i), int32), 255)
      s1 = mod(s1 + b, 65521)
      s2 = mod(s2 + s1, 65521)
    end do
    out = ior(ishft(s2, 16), s1)
  end function adler32

  integer(int32) function crc32(data, n) result(out)
    integer(int8), intent(in) :: data(:)
    integer, intent(in) :: n
    integer(int32) :: c, i, b, k, bit
    integer(int32), parameter :: poly = transfer(int(z'EDB88320', int64), 0_int32)

    c = not(0_int32)
    do i = 1, n
      b = iand(int(data(i), int32), 255)
      c = ieor(c, b)
      do k = 1, 8
        bit = iand(c, 1_int32)
        c = ishft(c, -1)
        if (bit /= 0) c = ieor(c, poly)
      end do
    end do
    out = not(c)
  end function crc32

end program poisson
