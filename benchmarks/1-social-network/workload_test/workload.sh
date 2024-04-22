front_end=$1
/home/azureuser/firm_compass/wrk2/wrk -D exp -t 4 -c 8 -d 60 -P compose_latencies.txt -L -s /home/azureuser/firm_compass/wrk2/scripts/social-network/compose-post.lua http://$front_end:31499/wrk2-api/post/compose -R 10 > compose.log
/home/azureuser/firm_compass/wrk2/wrk -D exp -t 4 -c 8 -d 60 -P user_latencies.txt -L -s /home/azureuser/firm_compass/wrk2/scripts/social-network/read-user-timeline.lua http://$front_end:31499/wrk2-api/user-timeline/read -R 30 > user.log
/home/azureuser/firm_compass/wrk2/wrk -D exp -t 4 -c 8 -d 60 -P home_latencies.txt -L -s /home/azureuser/firm_compass/wrk2/scripts/social-network/read-home-timeline.lua http://$front_end:31499/wrk2-api/home-timeline/read -R 60 > home.log
