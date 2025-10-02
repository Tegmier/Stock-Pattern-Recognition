import blpapi
from connection import test_bloomberg_connection, create_session
import datetime

ticker = "AAPL US Equity"
start_date = "20240101"
end_date = "20240501"

test_bloomberg_connection()
session = create_session()
session.start()

session.openService("//blp/refdata")
service = session.getService("//blp/refdata")
status = service.isValid()

request = service.createRequest("HistoricalDataRequest")
# request = service.createRequest("ReferenceDataRequest")

request.getElement("securities").appendValue(ticker)

request.getElement("fields").appendValue("BEST_SALES")
# request.getElement("fields").appendValue("ERN_ANN_DT")

request.set("startDate", start_date)
request.set("endDate", end_date)
request.set("periodicitySelection", "QUARTERLY")

session.sendRequest(request)

while True:
    ev = session.nextEvent()
    for msg in ev:
        print(msg)

    if ev.eventType() == blpapi.Event.RESPONSE:
            break

session.stop()
