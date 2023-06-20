
for top in */; do

    cd $top

    for d in */; do
        cd $d
    
        beamdyn_driver *.inp 
    
        cd ..

    done

    cd ..

done
