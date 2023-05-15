wrk.headers["Content-Type"]="application/json"
--token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZG1pbiIsInJvbGVzIjpbIlJPTEVfQURNSU4iXSwiaWQiOiJiYTA1ZDFiMS1hNDY1LTQzZjgtYmMyOC1mYmQ1NWIwYjQ3Y2UiLCJpYXQiOjE2MDU3NDczNjYsImV4cCI6MTYwNTc1MDk2Nn0.XXJ7-Lu-JXg4LXj5qipRHGgElLd2pi2os76EXtGl1Vc"
--wrk.headers["Authorization"]= "Bearer "..token
counter = 1
counter1 = 1
limit=1

stations = {}          -- create the matrix
function init(args)
  token=args[1]
  wrk.headers["Authorization"]= "Bearer "..token
  limit=args[2]
  for i=1,limit do
      stations[i] = {}     -- create a new row
      for j=1,limit do
        stations[i][j] = "station"..tostring(j)
      end
    end  
end 


request = function()  
       path  = "http://localhost:8080/api/v1/adminrouteservice/adminroute"
       local startStation = stations[counter][counter]
       local endStation = stations[counter][counter1]
       
       if counter1 == limit then 
               counter1=1
               counter=counter+1
       else 
          counter1=counter1+1
       end 
       --local distance = math.random(1,200)
       local body
       body ="{\"stationList\":\""..startStation..","..endStation.."\",\"distanceList\":\"0,100\",\"startStation\":\""..startStation.."\",\"endStation\":\""..endStation.."\"}" 
       --print(body)  
      return wrk.format("POST",path,headers,body)
   end
  
 response = function(status, headers, body)
      --print(body)
      if status == 200 then   
               -- print("success")
      else 
         --print("failed")          
     end
  end
