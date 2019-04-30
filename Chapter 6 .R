urlToRead <- "https://www2.census.gov/programs-surveys/popest/tables/2010-2011/state/totals/nst-est2011-01.csv "
testFrame <- read.csv(url(urlToRead))
str(testFrame)
summary(testFrame)

testFrame <- testFrame[,1:5]

head(testFrame[1],10)
testFrame <- testFrame[-1:-8,]
tail(testFrame[1],10)
testFrame <- testFrame[-52:-58,]

colnames(testFrame)
testFrame$stateName <- testFrame[,1]
cnames <- colnames(testFrame)
cnames[1] <-'newName'
cnames
colnames(testFrame)<-cnames
colnames(testFrame)
testFrame <- testFrame[,-1]
str(testFrame)

testFrame$stateName <- gsub("\\.","",testFrame$stateName)
testFrame$april110census <- gsub(",","",testFrame$X)
testFrame$april110base <- gsub(",","",testFrame$X.1)
testFrame$july10pop <- gsub(",","",testFrame$X.2)
testFrame$july11pop <- gsub(",","",testFrame$X.3)

str(testFrame)

testFrame$april110census <- as.numeric(gsub("","",testFrame$april110census))
testFrame$april110base <- as.numeric(gsub("","",testFrame$april110base))
testFrame$july10pop <- as.numeric(gsub("","",testFrame$july10pop))
testFrame$july11pop <- as.numeric(gsub("","",testFrame$july11pop))

str(testFrame)

testFrame<-testFrame[,-1:-4]
summary(testFrame)
str(testFrame)
head(testFrame)

rownames(testFrame) <-NULL
testFrame[c(1,3,5),]
sortedStates <- testFrame[order(testFrame$july11pop,decreasing = TRUE),]
sortedStates