Note: home-timeline doesn't have a DB of its own. So Redis is a DB and not a cache for it. That's why user-timeline-service has more variations.

compose_valid_json.1
text-service is the last service to finish so does the remaining tasks

compose_valid_json.2
unique_upload_id services is the last service

home_valid_json.1
home-timeline-service calls gets the post IDs from Redis. Passes this to post-storage-service. ReadPosts method of PSS checks if they are in memcached. If some or all or missing, gets them from mongo and sets the memcached too.

home_valid_json.2
same as 1 but all the posts are found in post-storage-memcached so no call to mongo.

user_valid_json.1
user-timeline-service checks for post IDs in Redis. If they are not found, gets them from mongo and updates Redis. Passes the post IDs to post-storage-service which searches for posts in memcached. checks for remaining posts, if any, in mongo and updates the memcached

user_valid_json.2
same as above but all post IDs are found in Redis. but PSS accesses mongo too.

user_valid_json.3
same as above but PSS also finds posts in memcached. So all cache access.

user_valid_json.4
user-timeline-service checks for post IDs in Redis. If they are not found, gets them from mongo and updates Redis. PSS finds them in memached.

