require "socket"
math.randomseed(socket.gettime()*1000)
math.random();

wrk.headers["Content-Type"]="application/json"
   --token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZG1pbiIsInJvbGVzIjpbIlJPTEVfQURNSU4iXSwiaWQiOiIzMTYwY2M1NS0yNjE3LTQzMGEtYmRhNi1kNTA4MWVmMTQ5YTMiLCJpYXQiOjE2MDMzOTgwNzYsImV4cCI6MTYwMzQwMTY3Nn0.UnID3i1LtonA
   counter = 1
   counter1= 1
   counter2 = 1
   search1=1
   search2=1
   limit=1
   travels = {}          -- create the matrix--trains = {}
   stations = {}
   nowTable = os.date('*t')
   Year = nowTable.year
   Month = nowTable.month
   stations={}
  
  function init(args)
    token=args[1]
    wrk.headers["Authorization"]= "Bearer "..token
    limit=args[2]
    for i=1,limit do
        stations[i]={}
        for j=1,limit do
            stations[i][j] = "station"..tostring(j)
        end     -- create a new row
     end
   end
  
--request = function()
     --print("request")  
         

local function search()
      local path="http://localhost:8080/api/v1/travelservice/trips/left"
      --local startStation = stations[search1][search1]
      --local endStation = stations[search1][search2]
      local startStation = "station"..tostring(math.random(1,limit))
      local endStation = "station"..tostring(math.random(1,limit))
      --local travelID = "Travel"..startStation..endStation
       --  counter = counter + 1
         if search2  == limit then
                  search2=1
                  search1=search1+1
           else
            search2=search2+1
          end
      local body
      body="{\"startingPlace\":\""..startStation.."\",\"endPlace\":\""..endStation.."\",\"departureTime\":\""..Year.."-"..Month.."-28\"}"
      --print(body)
      return wrk.format("POST",path,headers,body)
end
  

local function book()
    path  = "http://localhost:8080/api/v1/preserveotherservice/preserveOther"
	  --local startStation = stations[counter1][counter1]
	  --local endStation = stations[counter1][counter2]
      local startStation = "station"..tostring(math.random(1,limit))
      local endStation = "station"..tostring(math.random(1,limit))
	  local travelID = "Travel"..startStation..endStation
 counter = counter + 1
         if counter2 == limit then
                  counter2=1
                  counter1=counter1+1
          else
            counter2=counter2+1
          end
          local body
          body = "{\"accountId\":\"4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f\",\"contactsId\":\"3417cd74-fb4c-4002-b108-710ce71705e4\",\"tripId\":\""..travelID.."\",\"seatType\":\"2\",\"date\":\""..Year.."-"..Month.."-28\",\"from\":\""..startStation.."\",\"to\":\""..endStation.."\",\"assurance\":\"0\",\"foodType\":1,\"foodName\":\"Bone Soup\",\"foodPrice\":2.5,\"stationName\":\"\",\"storeName\":\"\"}"
        -- print(body)
        return wrk.format("POST",path,headers,body)

end

request = function()
  local search_ratio      = 0.5
  local book_ration   = 0.5
  

  local coin = math.random()
  if coin < search_ratio then
    return search()
  else
    return book()
  end
end

--response = function(status, headers, body)
  
     -- print(body)  
      --if status == 200 then
        --        print("success1")
    --    else
          -- print("failed")          
  --     end
--end
