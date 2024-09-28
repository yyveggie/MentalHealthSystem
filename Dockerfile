FROM ccr.ccs.tencentyun.com/aw16/python:3.10.15-slim-bookworm
# ccr.ccs.tencentyun.com/aw16/python:3.10.11-slim-buster
# ccr.ccs.tencentyun.com/aw16/python:3.11.4-slim-buster
# ccr.ccs.tencentyun.com/aw16/python:3.11.10-slim-bookworm
# ccr.ccs.tencentyun.com/aw16/python:3.10.15-slim-bookworm

RUN mkdir -p /etc/apt
#RUN wget http://mirrors.163.com/.help/sources.list.jessie -O /etc/apt/sources.list
#RUN sed -i "s@http://deb.debian.org/debian@http://mirror.sjtu.edu.cn/debian@g" /etc/apt/sources.list

RUN rm -f /etc/apt/sources.list
RUN echo "deb http://mirror.zju.edu.cn/debian buster main contrib non-free" > /etc/apt/sources.list
RUN echo "deb http://mirror.zju.edu.cn/debian buster-updates main contrib non-free" >> /etc/apt/sources.list
RUN echo "deb http://mirror.zju.edu.cn/debian-security buster/updates main contrib non-free" >> /etc/apt/sources.list

RUN apt-get update
RUN apt-get -f install
#RUN apt-get purge -y python
#RUN apt-get install -y python
# RUN apt-get install -y unzip vim python-dev libav-tools libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig libpulse-dev
RUN python --version
# ENV ATHENA_ENV test

# ENV PHANTOMJS_VERSION 2.1.1
# ENV PHANTOMJS_PLATFORM linux-x86_64

# add gitlab access token
RUN mkdir -p /root/.ssh/
RUN echo "-----BEGIN OPENSSH PRIVATE KEY-----\n\
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAABFwAAAAdzc2gtcn\n\
NhAAAAAwEAAQAAAQEAy8VHss8BmEi3eapyBh1lDzB1acjShvv0TeS2qFqJAjRwfI32HCd3\n\
GzTSJMcqlTxer0n2RlI5fYrLoJnc+3ovcoh9AhlSZhluTPmyytlbSf0HescYArRznjEUAT\n\
QTS2w1s5zMS3mSWBsikpkP/0tDo4K3AwZSPrK6HcC/bqJFofN0Xn2WWZAm5ijebBr2AfeD\n\
nQnymqj3VFfW4hfi8bosQ5quRgHYSKVRqW+hEMK58xbu0/kntwQtdPFj6co1QoljyNUESf\n\
nWOpTkigxngu/3lTGMw7DQPanoL01r2FFAST7a1xg/Xf1ic6WxTJj6K0qpofTfvj7W+LaB\n\
3aLXVye8nwAAA+BZAZq7WQGauwAAAAdzc2gtcnNhAAABAQDLxUeyzwGYSLd5qnIGHWUPMH\n\
VpyNKG+/RN5LaoWokCNHB8jfYcJ3cbNNIkxyqVPF6vSfZGUjl9isugmdz7ei9yiH0CGVJm\n\
GW5M+bLK2VtJ/Qd6xxgCtHOeMRQBNBNLbDWznMxLeZJYGyKSmQ//S0OjgrcDBlI+srodwL\n\
9uokWh83RefZZZkCbmKN5sGvYB94OdCfKaqPdUV9biF+LxuixDmq5GAdhIpVGpb6EQwrnz\n\
Fu7T+Se3BC108WPpyjVCiWPI1QRJ+dY6lOSKDGeC7/eVMYzDsNA9qegvTWvYUUBJPtrXGD\n\
9d/WJzpbFMmPorSqmh9N++Ptb4toHdotdXJ7yfAAAAAwEAAQAAAQEAgVqb+DuGtKg8qKR1\n\
u8H/PFQzxNJyKrRY0vXEqjzGyrqFSdxIhnjbGE9As44nTxNCvB3Ek8Fws2xfdXUXHiDja9\n\
6PFL0EXMH6Di1YvYKbe+1/SQaIDHeHCb34CAeJ1BA+SuXXnnXvW7TmDo/R4uthZbP9/5KM\n\
tTZ6BZctLsseTv5+G6KEh27edTehDnfxv5p2llYNDhI0MJTFOXLY4ALRMW1gBAPs6w2HTW\n\
eGjdmouUUkUbzfUhvQA4xVN1q96sgX8nHE/U1nvL1Cwc3SLD2W7XUst1KF4FJGdiKuYnhR\n\
g6GsPlc8RUPbeMjyxLKL+cR5sxz4XznrH2l99wwZPMUnEQAAAIEApIup0RyuVGhRUWvSDL\n\
2rDIGyv9MqgB4xrQ5lJa1r/CYLar475tdna2BxkfDcRoGuBmXTtyVezOgvtgiwvl1stQgy\n\
BoRZZvhXuliqwbmm/FPDEcf3IQCu9L8Lro7kk6xNnYWeciQH1NcyZySF3ukaN0igjwq+3I\n\
RW+x5AJxkSy3YAAACBAOuZRFVzBr4YFnziIZPLrmltfUTbt3stYyqMQAzzDqFE2XLtuYTF\n\
RxwmlSK+EksJ4vvkn2veKLzjpI4L63MRViitVNEOEqbcOIxWtQCzZ9oPCj1r0jNFBml+Jl\n\
nYaMSpzxxKTCXLS7LNygM097Cs1g2F3azKk+HyC/gc/hqCt2fJAAAAgQDdanNx+vnVLZvm\n\
1uqSMWLaZQftzuXBRlBRJyHXRPl/yDncQpRwJSNhXFckpz94wb+F4QMM6YHpwbHLtZcZu8\n\
GXbgUH7QhQkLhdmMCIWjysb7+WTRypsPgTIlf0tWFEpHgFfIAHHBe2DkdeCemC9v3iGEvG\n\
4N+fOJEEgnNfjrAFJwAAACRzb25namlhbkBzb25namlhbmRlTWFjQm9vay1Qcm8ubG9jYW\n\
wBAgMEBQ==\n\
-----END OPENSSH PRIVATE KEY-----" > /root/.ssh/id_rsa
RUN cat /root/.ssh/id_rsa

# RUN chmod 0600 /root/.ssh/id_rsa
# RUN touch /root/.ssh/known_hosts
# RUN apt-get install -y openssh-client
# RUN ssh-keyscan xy-gitlab.aw16.com >> /root/.ssh/known_hosts
# RUN ssh -T git@xy-gitlab.aw16.com

# Install Chrome
# RUN wget -N https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb -P ~/
# RUN dpkg -i --force-depends ~/google-chrome-stable_current_amd64.deb; exit 0
# RUN apt-get -f install -y
# RUN dpkg -i --force-depends ~/google-chrome-stable_current_amd64.deb

# Install ChromeDriver
# RUN wget https://npm.taobao.org/mirrors/chromedriver/2.35/chromedriver_linux64.zip -P ~/
# RUN unzip ~/chromedriver_linux64.zip -d ~/
# RUN rm ~/chromedriver_linux64.zip
# RUN mv -f ~/chromedriver /usr/local/share/
# RUN chmod +x /usr/local/share/chromedriver
# RUN ln -s /usr/local/share/chromedriver /usr/local/bin/chromedriver

# Install Phantomjs
# RUN wget -q -O /tmp/phantomjs-$PHANTOMJS_VERSION-linux-x86_64.tar.bz2 http://cdn.npm.taobao.org/dist/phantomjs/phantomjs-$PHANTOMJS_VERSION-linux-x86_64.tar.bz2 && \
#   tar -xjf /tmp/phantomjs-$PHANTOMJS_VERSION-linux-x86_64.tar.bz2 -C /tmp && \
#   rm -f /tmp/phantomjs-$PHANTOMJS_VERSION-linux-x86_64.tar.bz2 && \
#   mv /tmp/phantomjs-$PHANTOMJS_VERSION-linux-x86_64/ /usr/local/share/phantomjs && \
#   ln -s /usr/local/share/phantomjs/bin/phantomjs /usr/local/bin/phantomjs

WORKDIR /usr/src/app
#安装python依赖
ADD ./requirement.txt /usr/src/app/requirements.txt
RUN export PATH="/root/.local/bin:$PATH"
ENV PATH=/root/.local/bin:$PATH
#RUN pip install --upgrade pip
# RUN pip install pip==24.2
# RUN pip install incremental==17.5.0
#RUN pip install -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com -r requirements.txt
RUN pip install  --user -i https://pypi.mirrors.ustc.edu.cn/simple/ --trusted-host pypi.mirrors.ustc.edu.cn -r requirements.txt
RUN pip install protobuf==3.20 -i https://pypi.tuna.tsinghua.edu.cn/simple
#代码copy到app目录
ADD . /usr/src/app

RUN chmod +x ./entrypoint.sh
EXPOSE 8765
ENTRYPOINT [ "./entrypoint.sh" ]
