C ftn_wig6j.f -- stand-in for the WIG6J pass-through of src/wt9x.fpl.
C
C XWIG9X calls WIG6J, which in ATOMCI is a pass-through to XWIG6J in
C wt9x.o. This copy lets the L0/L1 tests link xwig6j.o without wt9x.o.
      FUNCTION WIG6J(I1,I2,I3,I4,I5,I6)
      IMPLICIT REAL*8(A-H,O-Z)
      WIG6J = XWIG6J(I1,I2,I3,I4,I5,I6)
      RETURN
      END
