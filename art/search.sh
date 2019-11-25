#! /bin/sh
set +e

FROM=$1
TO=$2

DATE1="2019.10.29"
DATE="2019-10-29"
REQ='{"query": {"range": {"@timestamp": {"gte": "'${DATE}'T'${FROM}':00.000Z", "lte": "'${DATE}'T'${TO}':59.999Z"}}}}' 
echo ${REQ}

es2csv -u http://localhost:9200 -i nginx-${DATE1} -S @timestamp -s 10000 -f @timestamp clientip host request_time method request_length resp_code bytes -r -q "${REQ}" -o ${FROM}.csv

