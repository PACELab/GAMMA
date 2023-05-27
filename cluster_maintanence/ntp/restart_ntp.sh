# Restarts the ntp service.

for i in {2..16}
do
	ssh -i ~/compass.key ubuntu@userv$i "sudo service ntp restart; ntpq -np"
done

ssh -i ~/compass.key ubuntu@userv17-jaeger "sudo service ntp restart; ntpq -np" 
