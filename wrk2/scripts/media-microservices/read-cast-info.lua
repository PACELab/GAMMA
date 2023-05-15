require "socket"
local time = socket.gettime()*1000
math.randomseed(time)
math.random(); math.random(); math.random()

request = function()
  local chosen_cast_id = tostring(1)
  local num_casts = tostring(math.random(10,12))
    
  local args = "cast_id=" .. chosen_cast_id .. "&num_casts=" .. num_casts
  local method = "GET"
  local headers = {}
  headers["Content-Type"] = "application/x-www-form-urlencoded"
  -- Use your cluster-ip here:
  --local path = "http://localhost:8080/wrk2-api/user-timeline/read?" .. args
  local path = "http://userv2:8080/wrk2-api/cast-info/read?" .. args
  return wrk.format(method, path, headers, nil)

end
