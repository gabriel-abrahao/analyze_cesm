
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "~/lib_abrahao.ncl" ;You can find this on ftp.biosfera.dea.ufv.br/gabriel/lib_abrahao.ncl


begin ;SCRIPT


;infolder		=	"input/sheffield/"
;outfolder		=	"saidas/sheffield/"
infolder		=	"../input/rcp_daily/rcp2.6_seg_005/run/"
outfolder		=	"../output/rainy_season/rcp2.6_seg_005/"

invarname		=	"PRECT"

prefix			=	"rcp2.6_seg_005.cam2.h1."
suffix			=	"-01-01-00000.nc"


; mskfile			=	"mascara_tudo.nc"
; mskvarname		=	"surta"

;syear			=	2009
;eyear			=	2049	;Keep in mind that we also need the year after this one to compute the "agricultural year"
syear			=	2005
eyear			=	2006	;Keep in mind that we also need the year after this one to compute the "agricultural year"

baseval			=	2.5

;outfname		=	"days_" + baseval + ".nc"
outfname		=	"tudo_allyears_" + baseval + "_" + syear + "_" + eyear + ".nc"
outmeanfname		=	"mean_" + syear + "_" + eyear + ".nc"

; daysyear		=	182	;Day that our ag year will start, considering nonleap
daysyear		=	180	;Day that our ag year will start, 360 day calendar

timeaxis		=	0	;0 time in ag year days, 1 time in actual DOY

nyears = eyear - syear + 1

;Checking dimensions to allocate 
dumfile = addfile(infolder + prefix + syear + suffix,"r")
dumvar = dumfile->$invarname$ 
dumdims = dimsizes(dumvar)


outoday = new(array_append_record(nyears,dumdims(1:2),0),typeof(dumvar),dumvar@_FillValue) ;Output variable for onset day
copy_VarCoords_exl(dumvar,outoday) ;You can find this function on ftp.biosfera.dea.ufv.br/gabriel/lib_abrahao.ncl
outoday!0 = "time"
outoday&time = fspan(syear,eyear,nyears) ;WARNING: Bugs if there is only one year
outoday&time@units = "year"
;outoday&time@units = "years since 00-00-0000"
;outoday&time@calendar = "proleptic_gregorian"
outeday = outoday ;Output variable for end day
outslen = outoday ;OUtput variable for season length

; Create output folder
system("mkdir -p " + outfolder)

system("rm " + outfolder + outfname)
arqout = addfile(outfolder + outfname,"c")
arqout@baseval = baseval
arqout@daysyear = daysyear

system("rm " + outfolder + outmeanfname)
arqoutmean = addfile(outfolder + outmeanfname,"c")
arqoutmean@baseval = baseval
arqoutmean@daysyear = daysyear

do year = syear,eyear
	print((/"Running year " + year/))
	arqin1 = addfile(infolder + prefix + year + suffix,"r")
	varin1 = arqin1->$invarname$
	arqin2 = addfile(infolder + prefix + (year+1) + suffix,"r") ;We must read the next year in order to change the starting date of our year
	varin2 = arqin2->$invarname$

	ndays1 = dimsizes(varin1&time) ;The number of days on the first year, to check for leap years
	ndays2 = dimsizes(varin2&time) ;The number of days on the first year, to check for leap years

	;isleap1 = isleapyear(year)
	;isleap2 = isleapyear(year + 1)
	isleap1 = ndays1.eq.366
	isleap2 = ndays2.eq.366



	if (.not.(isleap1 .or. isleap2) .or. (isleap2 .and. daysyear.lt.60) .or. (isleap1 .and. daysyear.ge.60))
		;365-day conditions
		if (.not.isleap1) then
			varin = varin1 
		else
			varin = varin2 
		end if
	else
		;366-day conditions
		if (isleap1) then
			varin = varin1 
		else
			varin = varin2 
		end if	
	end if; Cheking for year length

	varin = varin@_FillValue

	ndays = dimsizes(varin&time)
print((/ndays/))
	;Now we start building our ag year
	if (ndays .eq.365) then
		if (.not.(isleap1.or.isleap2)) then
			varin(:365-daysyear,:,:) = varin1(daysyear-1:,:,:)
			varin(365-daysyear+1:,:,:) = varin2(:daysyear-2,:,:)
		else if (isleap1 .and. daysyear.ge.60) then
			varin(:365-daysyear,:,:) = varin1(daysyear-1+1:,:,:)
			varin(365-daysyear+1:,:,:) = varin2(:daysyear-2,:,:)
		else if (isleap2 .and. daysyear.lt.60) then ;NOT TESTED
			varin(:365-daysyear,:,:) = varin1(daysyear-1:,:,:)
			varin(365-daysyear+1:,:,:) = varin2(:daysyear-2+1,:,:)

		end if
		end if
		end if

	else ;Case where ndays=366
		if (isleap1 .and. daysyear.lt.60) then ;NOT TESTED
			varin(:366-daysyear,:,:) = varin1(daysyear-1+1:,:,:)
			varin(366-daysyear+1:,:,:) = varin2(:daysyear-2,:,:)
		else if (isleap2 .and. daysyear.ge.60) then 
			varin(:365-daysyear,:,:) = varin1(daysyear-1:,:,:)
			varin(365-daysyear+1:,:,:) = varin2(:daysyear-2+1,:,:)
		
		end if
		end if


	end if

	if (timeaxis.eq.0) then
		varin&time=ispan(0,dimsizes(varin&time)-1,1)
	end if	

	; Convert units
	varin = varin*86400000.0
	varin@units = "mm/day"

	aaout = varin ;Anomalous accumulation
	aaout = 0.0
	aaout(0,:,:) = varin(0,:,:) - baseval ;First day of accumulation

	daaout = aaout ;Derivative of AA
	daaout = 0.0

	mdaa = daaout(0,:,:) ;Maximum value of dAA
	mdaaday = daaout(0,:,:) ;Day of maximum value of dAA
	mdaaday = mdaaday@_FillValue 

	minaa = aaout(0,:,:) ;Minimum value of AA before the maximum of dAA
	maxaa = aaout(0,:,:) ;Maximum value of AA after the maximum of dAA

	minaaday = minaa
	maxaaday = maxaa

	minaaday = minaaday@_FillValue
	maxaaday = minaaday@_FillValue

	mdaaday = mdaaday@_FillValue
	mdaa = -99999999.0 ;Very small number, negative

	;Metadata
	aaout@long_name = "Anomalous accumulation"
	daaout@long_name = "Derivative of anomalous accumulation"
	mdaa@long_name = "Maximum AA derivative"
	mdaaday@long_name = "Day of maximum AA derivative"
	minaa@long_name = "Minimum Anomalous accumulation"
	maxaa@long_name = "Maximum Anomalous accumulation"
	minaaday@long_name = "Day of minimum Anomalous accumulation"
	maxaaday@long_name = "Day of maximum Anomalous accumulation"
	

	do day = 1,ndays-1
		aaout(day,:,:) = aaout(day-1,:,:) + varin(day,:,:) - baseval
		daaout(day,:,:) = aaout(day,:,:) - aaout(day-1,:,:)

		dummdaa = where(daaout(day,:,:).gt.mdaa,daaout(day,:,:),mdaa)
		mdaaday = where(daaout(day,:,:).gt.mdaa,day,mdaaday)
		mdaa = (/dummdaa/)

	end do ;DAY

	;Average daily rainfall and mask if is more than baseval
	avgrain = dim_avg_n_Wrap(varin,0)
	basemsk = where(avgrain.lt.baseval,False,True)


	;This is the method for avgrain.lt.basval
	;FIXME: We have to loop again because we need the min(max) aa before(after) the maximum daa, and we don't know the max daa before finishing the loop, but there might be a workaround
	do day = 1,ndays-1
		dumminaa = where((aaout(day,:,:).lt.minaa).and.(day.lt.mdaaday),aaout(day,:,:),minaa)
		dummaxaa = where((aaout(day,:,:).gt.maxaa).and.(day.gt.mdaaday),aaout(day,:,:),maxaa)

		minaaday = where((aaout(day,:,:).lt.minaa).and.(day.lt.mdaaday),day,minaaday)
		maxaaday = where((aaout(day,:,:).lt.maxaa).and.(day.lt.mdaaday),day,maxaaday)

		minaa =(/dumminaa/)
		maxaa =(/dummaxaa/)
	end do ;DAY

	;Override values from the method above if its more than baseval
	minaa = where(basemsk,dim_min_n_Wrap(aaout,0),minaa)
	maxaa = where(basemsk,dim_max_n_Wrap(aaout,0),maxaa)


	do day = 1,ndays-1
		minaaday = where(aaout(day,:,:).eq.minaa,day,minaaday)
		maxaaday = where(aaout(day,:,:).eq.maxaa,day,maxaaday)

	end do

	;OLD:Output variables in actual DOY, FIXME: with maximum 1 day error and not going around the year
;	outoday(year-syear,:,:) = minaaday + daysyear
;	outeday(year-syear,:,:) = maxaaday + daysyear
;	outslen(year-syear,:,:) = outeday(year-syear,:,:) - outoday(year-syear,:,:)
;
;	outoday = where(outoday.gt.365,(outoday-365),outoday)
;	outeday = where(outeday.gt.365,(outeday-365),outeday)
	;NEW: Output variables in AG year DOY 
	outoday(year-syear,:,:) = minaaday 
	outeday(year-syear,:,:) = maxaaday 
	outslen(year-syear,:,:) = outeday(year-syear,:,:) - outoday(year-syear,:,:)


	; ;Load output mask
	; arqmsk = addfile(mskfile,"r")
	; msk = arqmsk->$mskvarname$


	; if dimsizes(dimsizes(msk)).eq.3 then
	; 	dummsk = msk(0,:,:)
	; else if dimsizes(dimsizes(msk)).eq.4 then
	; 	dummsk = msk(0,0,:,:)
	; else
	; 	dummsk = msk
	; end if
	; end if
	
	; delete(msk)
	; msk = dummsk
	; delete(dummsk)
printVarSummary(outeday)
; printVarSummary(msk)

	; outoday = mask(outoday,msk,1)
	; outeday = mask(outeday,msk,1)
	; outslen = mask(outslen,msk,1)

	arqout->outoday=outoday
	arqout->outeday=outeday
	arqout->outslen=outslen

	arqoutmean->outoday=dim_avg_n_Wrap(outoday,0)
	arqoutmean->outeday=dim_avg_n_Wrap(outeday,0)
	arqoutmean->outslen=dim_avg_n_Wrap(outslen,0)

	;system("rm " + outfolder + "anac-" + year + ".nc")
	;dumarq = addfile(outfolder + "anac-" + year + ".nc","c")
	;dumarq@year = year
	;dumarq@baseval = baseval
	;dumarq@daysyear = daysyear
	;dumarq->aaout = aaout
	;dumarq->daaout = daaout
	;dumarq->mdaa = mdaa
	;dumarq->mdaaday = mdaaday
	;dumarq->minaaday = minaaday
	;dumarq->maxaaday = maxaaday
	;dumarq->minaa = minaa
	;dumarq->maxaa = maxaa


	;system("rm " + outfolder + "estc-" + year + ".nc")
	;dumarq = addfile(outfolder + "estc-" + year + ".nc","c")
	;dumarq@year = year
	;dumarq@ybaseval = baseval
	;dumarq->mdaa = mdaa
	;dumarq->mdaaday = mdaaday
	;dumarq->minaaday = minaaday
	;dumarq->maxaaday = maxaaday
	;dumarq->minaa = minaa
	;dumarq->maxaa = maxaa

	
	


	delete(varin1)
	delete(varin2)
	delete(varin)
	delete(aaout)
	delete(daaout)
	delete(mdaa)
	delete(mdaaday)
end do ;YEAR





end ;SCRIPT

