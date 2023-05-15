require "socket"
math.randomseed(socket.gettime()*1000)
math.random(); math.random(); math.random()

local charset = {'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's',
  'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm', 'Q',
  'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D', 'F', 'G', 'H',
  'J', 'K', 'L', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', '1', '2', '3', '4', '5',
  '6', '7', '8', '9', '0'}

local decset = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}

local function stringRandom(length)
  if length > 0 then
    return stringRandom(length - 1) .. charset[math.random(1, #charset)]
  else
    return ""
  end
end

local function decRandom(length)
  if length > 0 then
    return decRandom(length - 1) .. decset[math.random(1, #decset)]
  else
    return ""
  end
end

token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZG1pbiIsInJvbGVzIjpbIlJPTEVfQURNSU4iXSwiaWQiOiJmYjBjOTZlMy0yNzNlLTQ5YmMtOGQ2My01Yjg3MjQ1ZGFmNDYiLCJpYXQiOjE2MDIyNzMyNjQsImV4cCI6MTYwMjI3Njg2NH0.22DBqK9EDHqNhfjl44loPcUfipHSZAgAGc3g96cBKkM"
wrk.headers["Content-Type"]="application/json"
wrk.headers["Authorization"]= "Bearer "..token
  
   request = function()
   --print("request")  
       path  = "http://localhost:8080/api/v1/adminuserservice/users"
       local gender = math.random(0,2)
       local username = stringRandom(8)
       local password = stringRandom(8)
       local email = stringRandom(8)
       email = email.."@test.com"
       print(wrk.headers["Authorization"])
       local body
       body ="{\"userName\":\""..username.."\",\"password\":\""..password.."\",\"gender\":\""..gender.."\",\"email\":\""..email.."\"}" 
       print(body)
       return wrk.format("POST",path,headers,body)
   end
  
  response = function(status, headers, body)
      print(status)
      print(body)      
      if status == 200 then   
                print("success1")
      else 
         print("failed")          
     end
  end
