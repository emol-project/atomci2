C ftn_helper.f -- Fortran side of the L1 test (C port, stage 1).
C
C Linked together with objects from the M0 scratch build of src/
C (xcfp.o, setcfp.o, cfpset.o, isp.o, ior.o and the L0 set) so the C
C code can be compared with the Fortran in the same process.
C
C FTNLOAD(PATH) opens CFPTBL as unit 10 and fills /COMCFP/, /COMWIG/
C and /COMWFA/ the way ATOMCI does (SETCFP, SETWIG, CFPSET).
      SUBROUTINE FTNLOAD(PATH)
      CHARACTER*(*) PATH
      OPEN(10, FILE=PATH, FORM='UNFORMATTED', STATUS='OLD')
      CALL SETCFP(10)
      CLOSE(10)
      CALL SETWIG
      CALL CFPSET
      RETURN
      END
