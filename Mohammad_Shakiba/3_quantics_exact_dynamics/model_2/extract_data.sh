cat output | grep 'state = 1  pop' | awk '{print $5}' > pop_1.txt
cat output | grep 'state = 2  pop' | awk '{print $5}' > pop_2.txt
cat output | grep 'state = 2  pop' | awk '{print $11}' > ene_2.txt
cat output | grep 'state = 1  pop' | awk '{print $11}' > ene_1.txt
cat output | grep '<q>' | awk '{print $4}' > q.txt
