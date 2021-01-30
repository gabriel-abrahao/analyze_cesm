module dpres
contains
      subroutine dpres1d(klvl,plevel,psfc,pmsg,ptop,dp,iopt,kflag,ier) 
      implicit none
!                                                ! input 
      integer              klvl, iopt, kflag, ier
      double precision     plevel(klvl), psfc, ptop, pmsg
!                                                ! output
      double precision     dp(klvl)
!
! isobaric (constant) pressure level equivalent of dpres_hybrid_ccm
! The returned 'dp' are equivalent to having used  beta factors.
!                                                ! local
      integer             mono, kl, klStrt, klLast
      double precision    plvl(klvl), work(klvl)
      double precision    dpsum, pspan, peps, plow, phi

      peps  = 0.001d0
!
! check to see if ptop is reasonable
! .   psfc = pmsg   is not allowed
! .   ptop < 0      is not allowed
! .   ptop > psfcmx is probably due to units difference

      ier    = 0
      if (psfc.eq.pmsg .or. psfc.lt.0.0d0) ier = ier + 100
      if (ptop.lt.0.0d0) ier = ier + 1
      if (ptop.ge.psfc)  ier = ier + 10

! if ier.ne.0; input error with psfc and/or ptop

      if (ier.ne.0) then   
          kflag = 1
          do kl=1,klvl 
             dp(kl)   = pmsg
          end do
          dpsum = pmsg
          return
      end if

! monotonically increasing or decreasing? Code wants top to bottom
! if decreasing pressure make increasing; then flip back 

      if (plevel(2).gt.plevel(1)) then
          mono =  1
          do kl=1,klvl
             plvl(kl) = plevel(kl)
          end do
      else
          mono = -1
          do kl=1,klvl
             plvl(kl) = plevel(klvl-kl+1)
          end do
      end if

! initialize to missing

      do kl=1,klvl
         dp(kl) = pmsg                 
      end do

! calculate 'dp'; check if dpsum.eq.(psfc-ptop) within peps then return

      if (ptop.le.plvl(1) .and. psfc.ge.plvl(klvl)) then
          kflag = 0

          dp(1) = (plvl(1)+plvl(2))*0.5d0 - ptop
          do kl=2,klvl-1 
             dp(kl)= 0.5d0*(plvl(kl+1) - plvl(kl-1))
          end do
          dp(klvl) = psfc -(plvl(klvl)+plvl(klvl-1))*0.5d0

      else 
          kflag  = 1

! The klvl pressure levels in plvl define (klvl-1) layers.  There is
!  one fewer layer than pressure levels.
! Find klStrt and klLast so they define the smallest possible 
!  interval [plev(klLast), plev(klStrt)] that contains all 
!  layer mid-points within the interval [ptop, psfc].
!
! Starting with the bottom and moving up, find the first layer whose
!  midpoint is at or above ptop. klStrt is the bottom of this layer. 
          do klStrt=klvl,2,-1
             if ((plvl(klStrt-1)+plvl(klStrt))/2.lt.ptop) then
                 exit
             end if
          end do

! Starting the top layer and moving downward, find the first layer whose
! midpoint is at or below psfc. klLast is the top of this layer. 
          do klLast=1,klvl-1
             if ((plvl(klLast+1)+plvl(klLast))/2.gt.psfc) then
                 exit
             end if
          end do

!debugprint *,"klStrt=",klStrt," klLast=",klLast," ptop=",ptop
!debugprint *,"plvl(klStrt)=",plvl(klStrt)," plvl(klLast)=",plvl(klLast)
!debugprint *,"plvl(klStrt  )=",plvl(klStrt)  
!debugprint *,"plvl(klStrt+1)=",plvl(klStrt+1)
!debugprint *,"dp(klStrt)=",dp(klStrt)

          if (klStrt.eq.klLast) then
              dp(klStrt) = psfc-ptop
          elseif (klStrt.lt.klLast) then
              dp(klStrt) = (plvl(klStrt)+plvl(klStrt+1))*0.5d0 - ptop
              do kl=klStrt+1,klLast-1 
                 dp(kl)= 0.5d0*(plvl(kl+1) - plvl(kl-1))
              end do
              dp(klLast) = psfc -(plvl(klLast)+plvl(klLast-1))*0.5d0

! c c     else     ! klStrt>klLast  is a pathological case
! c c              ! plev(?)=500  plev(?+1)=550, psfc=540, ptop=510
! c c              ! both level are *between* levels
          end if

      end if

! error check

! f90 dpsum = sum(dplvl, 1, dplvl.ne.pmsg)  
      dpsum = 0.0d0
      do kl=1,klvl
         if (dp(kl).ne.pmsg) then
             dpsum = dpsum + dp(kl)
         end if
      end do

      pspan = psfc-ptop

      if (dpsum.gt.0.0d0 .and. pspan.ne.dpsum) then
          plow = pspan-peps 
          phi  = pspan+peps 
          if (dpsum.gt.phi .or. dpsum.lt.plow) then
              ier = -1
              do kl=1,klvl
                 dp(kl) = pmsg                 
              end do
          end if
      end if

! if necessary return to original order

      if (mono.lt.0) then
          do kl=1,klvl
             work(kl) = dp(kl)
          end do

          do kl=1,klvl
             dp(kl) = work(klvl-kl+1)
          end do
       end if

      return
      end
end module dpres
