#####################################################################
##        readState function setup for IST 687
#####################################################################
readStates <- function(){
 
  # https vs http
  # inspired by https://www.pauloldham.net/importing-csv-files-into-r/
  #  section: Reading a .csv from the web
  urlRemote <- "https://www2.census.gov/"
  path <- "programs-surveys/popest/tables/2010-2011/state/totals/"
  fileName   <- "nst-est2011-01.csv"
  urlToRead <- paste0(urlRemote, path, fileName)
  dfStates<- read.csv(url(urlToRead))
  
  #Remove empty columns:
  dfStates <- dfStates[,1:5]
  
  #Remove top 8 rows:
  dfStates <- dfStates[-1:-8,]
  rownames(dfStates)<-NULL
  
  #Remove bottom 6 rows:
  dfStates <- dfStates[-52:-58,]
  
  #Rename column, remove the old column, and normoalize it:
  dfStates$stateName <- dfStates[,1]
  dfStates <- dfStates[,-1]
  dfStates$stateName <- gsub("\\.","",dfStates$stateName)
  
  #Normalize X,X.1,X.2,X.3 variables:
  dfStates$base2010 <- gsub("\\,","",dfStates$X)
  dfStates$base2010 <- as.numeric(dfStates$base2010)
  
  dfStates$base2011 <- gsub("\\,","",dfStates$X.1)
  dfStates$base2011 <- as.numeric(dfStates$base2011)
  
  dfStates$Jul2010 <- gsub("\\,","",dfStates$X.2)
  dfStates$Jul2010 <- as.numeric(dfStates$Jul2010)
  
  dfStates$Jul2011 <- gsub("\\,","",dfStates$X.3)
  dfStates$Jul2011 <- as.numeric(dfStates$Jul2011)

  #Remove old X, X.1, X.2, X.3 columns:
  dfStates <- dfStates[,-1:-4]
  
  return(dfStates)
}


