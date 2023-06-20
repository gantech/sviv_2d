
for d in */; do
    cd $d

    beamdyn_driver *.inp 

    cd ..

done
