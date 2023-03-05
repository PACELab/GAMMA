#! /usr/bin/python3

import collections
import sys,json,time,subprocess,copy
import requests
import random
import os

listOfApps = ["SN","HR","MM"]
endToStartInUsecs = (10*60*60*1000*1000)
numTracesToLookupPerIter = 1500

servNameLookup = {
    'compose': 'nginx-web-server',
    'home' :  'nginx-web-server',
    'user' : 'nginx-web-server' ,

    'reviewCompose' : 'compose-review-service',
    'reviewRead' : 'movie-review-service&operation=ReadMovieReviews',
    'cast' : 'cast-info-service&operation=ReadCastInfo',
    'plot' : 'plot-service',
}

operation_name_lookup = {
    'compose' : '/wrk2-api/post/compose',
    'home' : '/wrk2-api/home-timeline/read',
    'user' : '/wrk2-api/user-timeline/read'  
}

def getTraceIDs(endTime, startTime, storeDataFolderName, storeDataFilename,reqType,monMachine):
    serviceToQuery = servNameLookup[reqType] 
    allReqsData = []    

    urlPrefix = "http://"+str(monMachine)+":16686/api/traces?limit=10000&end="+str(endTime)+"&start="+str(startTime)
    urlSuffix = "&operation="+operation_name_lookup[reqType] + "&service="+str(serviceToQuery)
    url = str(urlPrefix)+str(urlSuffix)

    print ("\t url: %s "%(url))
    res = requests.get(url)

    traceIDs = []
    jdata = json.loads(res.content.decode('utf-8'))
    traces = jdata['data']
    print ("\t len(traces): %d "%(len(traces)))
    
    if len(traces) == 0:
        print("Stopping as the number of traces is 0")
        
    for curTrace in traces:
        traceIDs.append(curTrace['traceID']) 
        allReqsData.append(curTrace)

    data = {"data":allReqsData}
    
    output_folder = str(storeDataFolderName)
    os.system("mkdir -p %s" % output_folder)
    outputFile = output_folder + "/" + storeDataFilename
    with open(outputFile, "w+") as storeDataFile:
        json.dump(data, storeDataFile)

    print("\t Data stored in file: %s "%(outputFile))
    return data

def main(args):
    fileName = args[1]
    hostName = args[2]
    reqType = args[3]
    folderName = args[4]
    startTime = args[5]
    endTime = args[6]
    data = getTraceIDs(endTime, startTime, folderName, fileName, reqType, hostName)
    print("Length of traces captured ", len(data))

if __name__ == "__main__":
    main(sys.argv)
