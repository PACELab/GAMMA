wrk.headers["Content-Type"]="application/json"
--token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZG1pbiIsInJvbGVzIjpbIlJPTEVfQURNSU4iXSwiaWQiOiIzMTYwY2M1NS0yNjE3LTQzMGEtYmRhNi1kNTA4MWVmMTQ5YTMiLCJpYXQiOjE2MDMzOTgwNzYsImV4cCI6MTYwMzQwMTY3Nn0.UnID3i1LtonA
counter = 1
counter1= 1
counter2 = 1
limit=1
travels = {}          -- create the matrix
trains = {}
stations = {}
 
function init(args)
  token=args[1]
  wrk.headers["Authorization"]= "Bearer "..token
  limit=args[2]
  for i=1,limit do
      stations[i] = {}
      trains[i] = "Train"..tostring(i)
      --travels[i] = "Travel"..tostring(i)
      for j=1,limit do
          stations[i][j] = "station"..tostring(j)
      end     -- create a new row
   end  
 end 
   
request = function()
   --print("request")  
       path  = "http://localhost:8080/api/v1/admintravelservice/admintravel"
       local startStation = stations[counter1][counter1]
       local endStation = stations[counter1][counter2]
       local travelID = "Travel"..startStation..endStation
       local trainTypeId = trains[counter]
       local routeId = startStation..endStation  
       counter = counter + 1        
       --local distance = math.random(1,200)
       --print(travelID)
       --print(trainTypeId)
       if counter2 == limit then
               counter2=1
               counter1=counter1+1
       else 
         counter2=counter2+1
       end
       local body
       body= "{\"tripId\":\""..travelID.."\",\"trainTypeId\":\""..trainTypeId.."\",\"routeId\":\""..routeId.."\",\"startingTime\":\"1356762200000\"}"
       --pint(body)
       return wrk.format("POST",path,headers,body)
   end
  
  response = function(status, headers, body)
     
    --rint(body)  
    if status == 200 then   
                --print("success1")
      else 
        -- print("failed")          
     end
  end
