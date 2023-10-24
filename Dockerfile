FROM python:3.9-slim as builder
WORKDIR /usr
RUN mkdir /usr/local/java
ADD jdk-8u371-linux-x64.tar.gz /usr/local/java
RUN ln -s /usr/local/java/jdk1.8.0_371 /usr/local/java/jdk
ENV JAVA_HOME /usr/local/java/jdk
ENV JRE_HOME ${JAVA_HOME}/jre
ENV CLASSPATH .:${JAVA_HOME}/lib:${JRE_HOME}/lib
ENV PATH ${JAVA_HOME}/bin:$PATH
COPY . /usr
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
CMD ["python", "gpt3.py"]