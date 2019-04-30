#create Numberize() function - Gets rid of commas and other junk
#and converts to numbers

#Assume that the inputVector is a list of data that can be treated
#as character string
Numberize <- function(inputVector) {
  #Get rid of commas
  inputVector <- gsub(",","",inputVector)
  #Get rid of spaces
  inputVector <- gsub(" ","",inputVector)
  return(as.numeric(inputVector))
}

# Customized Function - read in the census data set
readCensus <- function() {
  #Assign a url variable
  # https vs http
  # inspired by https://www.pauloldham.net/importing-csv-files-into-r/
  #  section: Reading a .csv from the web
  urlRemote <- "https://www2.census.gov/"
  path <- "programs-surveys/popest/tables/2010-2011/state/totals/"
  fileName   <- "nst-est2011-01.csv"
  urlToRead <- paste0(urlRemote, path, fileName)

  #read the data from the web
  testFrame <- read.csv(url(urlToRead))
  
  #Keep the fisrt 5 columns which has data
  testFrame <- testFrame[,1:5]
  #remove the first 8 rows ('header information')
  testFrame <- testFrame[-1:-8,]
  #remove the last rows (tail info)
  testFrame <- testFrame[-52:-58,]
  
  #rename the first column and add it to the dataset
  testFrame$stateName <- testFrame[,1]
  #remove the first column
  testFrame <- testFrame[,-1]
  
  #remove the 'dot' from the state name
  testFrame$stateName <-gsub("\\.","",testFrame$stateName)
  
  #convert the columns to actual numbers and rename columns
  testFrame$april110census <- Numberize(testFrame$X)
  testFrame$april110base <- Numberize(testFrame$X.1)
  testFrame$july10pop <- Numberize(testFrame$X.2)
  testFrame$july11pop <- Numberize(testFrame$X.3)
  testFrame <- testFrame[,-1:-4]
  
  #remove old rownames, which are now confusing
  rownames(testFrame) <- NULL
  return(testFrame)
}
