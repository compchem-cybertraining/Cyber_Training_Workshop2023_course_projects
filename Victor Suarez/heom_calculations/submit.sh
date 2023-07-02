#!/bin/bash/

for i in 0.0 5.0 10.0 15.0 20.0

do
    DIR=/panasas/scratch/grp-cyberwksp21/vasuarez/heom_calculations/E0_${i}/
    cd ${DIR}
    sbatch ./job_run
done

