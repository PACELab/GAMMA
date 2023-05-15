local json = require('cjson')
token = ""
path  = "http://localhost:8080/api/v1/users/login"

--function init(args)
  --print(args[1])
  --print(args[2])
--end

request = function()
    local username = "admin"
    local password = "222222"
    local body = '{"username": "admin","password": "222222"}'
    local headers = {}
    headers["Content-Type"]="application/json"
    return wrk.format("POST",path,headers,body)
end

response = function(status, headers, body)
   if token=="" and status == 200 then
      --print("success")
      headers["Content-Type"]="application/json"
      --print(headers)
      --print(body)
      local tab = json.decode(body)
      token = tab["data"]["token"]
      local file = io.open("/home/ubuntu/uservices/uservices-perf-analysis/admin_token.txt", "w")
      file:write(token)
      file:close()
      print(token)
   end
end
