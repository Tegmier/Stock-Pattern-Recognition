import blpapi
from blpapi import SessionOptions, Session

# 配置 Session
options = SessionOptions()
options.setServerHost("localhost")
options.setServerPort(8194)

session = Session(options)

# 启动 Session
if not session.start():
    print("❌ 无法启动 Session")
    exit()

# 打开服务
if session.openService("//blp/refdata"):
    print("✅ 成功打开 //blp/refdata 服务")
    service = session.getService("//blp/refdata")
    if service.isValid():
        print("✅ Service 是有效的")
    else:
        print("❌ Service 无效")
else:
    print("❌ 无法打开 //blp/refdata 服务")
    session.stop()