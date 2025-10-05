! GPU-Accelerated Random Forest Core Routines
! Based on original Breiman-Cutler Fortran code with OpenACC directives
! Copyright (C) 2025 - GPU Acceleration additions
! Original code: Copyright (C) 2001-7 Leo Breiman and Adele Cutler

module rf_core
    implicit none

contains

    ! GPU-accelerated best split finding routine
    ! This is the compute-intensive kernel that benefits most from parallelization
    subroutine findbestsplit_gpu(x, y, classes, n, p, nclass, &
                                 ndstart, ndend, classpop, &
                                 msplit, bestsplit, decsplit, jstat, mtry)
        implicit none

        ! Input parameters
        integer, intent(in) :: n, p, nclass, ndstart, ndend, mtry
        real(8), intent(in) :: x(n, p)
        real(8), intent(in) :: y(n)
        integer, intent(in) :: classes(n)
        real(8), intent(in) :: classpop(nclass)

        ! Output parameters
        integer, intent(out) :: msplit, jstat
        real(8), intent(out) :: bestsplit, decsplit

        ! Local variables
        integer :: i, j, k, mt, mvar, nsp, nc
        real(8) :: crit0, critmax, crit
        real(8) :: pno, pdo, rln, rld, rrn, rrd
        real(8) :: wl(nclass), wr(nclass)
        real(8) :: xrand
        integer :: mind(p)
        integer :: nn, ntie

        ! Compute initial Gini criterion
        pno = 0.0d0
        pdo = 0.0d0
        !$ACC PARALLEL LOOP REDUCTION(+:pno,pdo)
        do j = 1, nclass
            pno = pno + classpop(j) * classpop(j)
            pdo = pdo + classpop(j)
        end do
        !$ACC END PARALLEL LOOP

        crit0 = pno / pdo
        jstat = 0
        critmax = -1.0d25

        ! Initialize variable indices for sampling
        do k = 1, p
            mind(k) = k
        end do
        nn = p

        ! Main loop: sample mtry variables and find best split
        ! This loop iterates over randomly selected features
        do mt = 1, mtry
            ! Random sampling without replacement
            call random_number(xrand)
            j = int(nn * xrand) + 1
            mvar = mind(j)
            mind(j) = mind(nn)
            mind(nn) = mvar
            nn = nn - 1

            ! For numerical predictors, evaluate all possible splits
            ! GPU acceleration: parallelize over split candidates
            rrn = pno
            rrd = pdo
            rln = 0.0d0
            rld = 0.0d0

            wl(:) = 0.0d0
            wr(:) = classpop(:)

            ntie = 1

            ! This is the critical loop for GPU acceleration
            ! Evaluate each potential split point
            !$ACC DATA COPYIN(x, classes, y) COPY(wl, wr)
            !$ACC PARALLEL LOOP PRIVATE(nc, k, crit) &
            !$ACC& REDUCTION(max:critmax)
            do nsp = ndstart, ndend - 1
                nc = nsp
                k = classes(nc)

                ! Update left and right Gini numerators/denominators
                rln = rln + y(nc) * (2.0d0 * wl(k) + y(nc))
                rrn = rrn + y(nc) * (-2.0d0 * wr(k) + y(nc))
                rld = rld + y(nc)
                rrd = rrd - y(nc)

                wl(k) = wl(k) + y(nc)
                wr(k) = wr(k) - y(nc)

                ! Check if split is valid (neither node empty)
                if (min(rrd, rld) > 1.0d-5) then
                    crit = (rln / rld) + (rrn / rrd)

                    if (crit > critmax) then
                        bestsplit = dble(nsp)
                        critmax = crit
                        msplit = mvar
                        ntie = 1
                    end if
                end if
            end do
            !$ACC END PARALLEL LOOP
            !$ACC END DATA
        end do

        if (critmax < -1.0d10 .or. msplit == 0) jstat = -1
        decsplit = critmax - crit0

    end subroutine findbestsplit_gpu


    ! Data partitioning routine
    ! Moves data based on split decision
    subroutine movedata_gpu(x, n, p, ndstart, ndend, msplit, &
                           splitval, indices_left, nleft)
        implicit none

        integer, intent(in) :: n, p, ndstart, ndend, msplit
        real(8), intent(in) :: x(n, p), splitval
        integer, intent(out) :: indices_left(n), nleft

        integer :: i, idx

        nleft = 0

        ! GPU-accelerated data partitioning
        !$ACC DATA COPYIN(x, splitval) COPYOUT(indices_left, nleft)
        !$ACC PARALLEL LOOP REDUCTION(+:nleft)
        do i = ndstart, ndend
            if (x(i, msplit) <= splitval) then
                nleft = nleft + 1
                indices_left(nleft) = i
            end if
        end do
        !$ACC END PARALLEL LOOP
        !$ACC END DATA

    end subroutine movedata_gpu


    ! Simple tree building with GPU acceleration
    ! Simplified version for demonstration
    subroutine build_tree_gpu(x, y, classes, n, p, nclass, &
                              max_depth, min_samples_split, &
                              tree_nodes, n_nodes)
        implicit none

        integer, intent(in) :: n, p, nclass, max_depth, min_samples_split
        real(8), intent(in) :: x(n, p)
        real(8), intent(in) :: y(n)
        integer, intent(in) :: classes(n)
        integer, intent(out) :: n_nodes
        real(8), intent(out) :: tree_nodes(1000, 5)  ! Simplified tree storage

        integer :: current_depth, msplit, jstat
        real(8) :: bestsplit, decsplit
        real(8) :: classpop(nclass)
        integer :: mtry

        ! Initialize
        n_nodes = 0
        current_depth = 0
        mtry = int(sqrt(real(p)))

        ! Compute class populations
        classpop(:) = 0.0d0
        !$ACC PARALLEL LOOP
        do i = 1, n
            classpop(classes(i)) = classpop(classes(i)) + 1.0d0
        end do
        !$ACC END PARALLEL LOOP

        ! Build root node with GPU acceleration
        call findbestsplit_gpu(x, y, classes, n, p, nclass, &
                              1, n, classpop, &
                              msplit, bestsplit, decsplit, jstat, mtry)

        if (jstat /= -1) then
            n_nodes = 1
            tree_nodes(1, 1) = dble(msplit)
            tree_nodes(1, 2) = bestsplit
            tree_nodes(1, 3) = decsplit
        end if

    end subroutine build_tree_gpu

end module rf_core
