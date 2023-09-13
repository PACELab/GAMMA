cmd=$1
for i in {2..16}
do
	echo userv$i
	ssh -i ~/compass.key ubuntu@userv$i "$cmd"
	echo "\n\n"

done
echo userv17-jaeger
ssh -i ~/compass.key ubuntu@userv17-jaeger "$cmd"
