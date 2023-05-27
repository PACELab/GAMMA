for i in {2..16}
do
	echo userv$i
	ssh -i ~/compass.key ubuntu@userv$i "cd /home/ubuntu/firm_compass/; git pull"
	echo "\n\n"

done
echo userv17-jaeger
ssh -i ~/compass.key ubuntu@userv17-jaeger "cd /home/ubuntu/firm_compass/; git pull"
