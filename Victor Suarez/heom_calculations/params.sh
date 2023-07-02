#!/bin/bash/

for i in 0.0 5.0 10.0 15.0  20.0

do
    DIR=/panasas/scratch/grp-cyberwksp21/vasuarez/heom_calculations/E0_${i}/
    DIR2=/panasas/scratch/grp-cyberwksp21/vasuarez/heom_calculations/
    cd ${DIR}
    cp $DIR2/job_libra.py .
    sed -i 's/delt_sub/0.5/g' job_libra.py
    sed -i 's/nsteps_sub/500000/g' job_libra.py
    sed -i 's/step_switch_sub/125000/g' job_libra.py
    sed -i "s/E0_sub/$i/g" job_libra.py
    sed -i 's/K_sub/16/g' job_libra.py
    sed -i 's/L_sub/1/g' job_libra.py
 
done

