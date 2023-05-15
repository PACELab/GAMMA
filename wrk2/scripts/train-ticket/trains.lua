wrk.headers["Content-Type"]="application/json"
--token = ""
--wrk.headers["Authorization"]= "Bearer "..token
counter = 1
counter1 = 1
limit=1 
trains = {}  

function init(args)
  token=args[1]
  wrk.headers["Authorization"]= "Bearer "..token
  limit=args[2]
  for i=1,limit do
      trains[i] = "Train"..tostring(i)
      --print(trains[i])
      counter1 = counter1+1
  end
end

  
request = function()
   --print("request")  
       path  = "http://localhost:8080/api/v1/adminbasicservice/adminbasic/trains"
       --local TrainID = trains[counter]
       local TrainID = "Train"..tostring(counter)
       counter=counter+1
       local body
       body="{\"id\":\""..TrainID.."\",\"economyClass\":2147483647,\"confortClass\":2147483647,\"averageSpeed\":120}"
       --print(body)
       return wrk.format("POST",path,headers,body)
   end
  
response = function(status, headers, body)
      --print(body)
     
      if status == 200 then   
            --print("success")
      else 
        -- print("failed")          
     end
 end
