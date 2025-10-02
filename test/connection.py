import blpapi
from blpapi import SessionOptions, Session

def test_bloomberg_connection():
    options = SessionOptions()
    options.setServerHost('localhost')  # 默认主机
    options.setServerPort(8194)         # 默认端口

    session = Session(options)

    if not session.start():
        print("无法启动 Bloomberg Session")
        return

    if not session.openService("//blp/refdata"):
        print("无法打开 Bloomberg 服务")
        return

    print("成功连接到 Bloomberg API")
    session.stop()

def create_session():
    options = SessionOptions()
    options.setServerHost('localhost')  # 默认主机
    options.setServerPort(8194)         # 默认端口

    session = Session(options)
    return session


# # 测试连接
# test_bloomberg_connection()