list:
	@echo "sync-codes: 同步代码"
	@echo "build: 编译可执行文件"
	@echo "test: 测试"

sync-codes:
	git pull

deps:
	echo "deps"
	@if [ "$(CI_COMMIT_REF_NAME)" = "dev" ]; then\
		echo "checkout ivankastd:dev";\
		git clone git@gitlab.itingluo.com:backend/deploy.git /tmp/govendor_temp;\
	elif [ "$(CI_COMMIT_REF_NAME)" = "tl-sit" ]; then\
    	echo "checkout ivankastd:sit";\
		git clone git@gitlab.itingluo.com:backend/deploy.git /tmp/govendor_temp;\
	else\
		echo "checkout ivankastd:tags";\
		git clone git@gitlab.itingluo.com:backend/deploy.git /tmp/govendor_temp;\
	fi
	mkdir -p /go/src/gitlab.wallstcn.com/$(CI_PROJECT_NAMESPACE)/
    cp -R "/tmp/govendor_temp/agent" "/go/src/xy-gitlab.aw16.com/backend/$(SERVICE_NAME)/"
  
build:
	go build -o $(SERVICE_NAME)

test:
	go test
