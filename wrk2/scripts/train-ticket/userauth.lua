local json = require('cjson')
token = ""
path  = "http://localhost:8080/api/v1/users/login"
 
request = function()

      --nowTable = os.date('*t')
      --nowYear = nowTable.year
      --nowHour = nowTable.hour
      --print(nowYear)      
      local username = "fdse_microservice"
      local password = "111111"
      local body = '{"username": "fdse_microservice","password": "111111"}'
      local headers = {}
      headers["Content-Type"]="application/json"
      return wrk.format("POST",path,headers,body)
end
  
response = function(status, headers, body)
     if token=="" and status == 200 then
     
        headers["Content-Type"]="application/json"
        local tab = json.decode(body)
        token = tab["data"]["token"]
        -- print(token)
        local file = io.open("/home/ubuntu/uservices/uservices-perf-analysis/user_token.txt", "w")
        file:write(token)
        file:close()
     end
end
