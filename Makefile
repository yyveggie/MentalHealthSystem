list:
	@echo "sync-codes: 同步代码"
	@echo "build: 编译可执行文件"
	@echo "test: 测试"

sync-codes:
	git pull

build:

deps:
	mkdir -p /tmp/govendor/src/gitlab.wallstcn.com/backend
	@if [ "$(CI_COMMIT_REF_NAME)" = "tl-sit" ]; then\
		echo "checkout ivankastd:tl-sit";\
		git clone git@gitlab.itingluo.com:backend/deploy.git /tmp/govendor_temp; \
	else\
		echo "checkout ivankastd:tags";\
		git clone git@gitlab.itingluo.com:backend/deploy.git /tmp/govendor_temp; \
	fi
	pwd
	ls
	cp -r /tmp/govendor_temp/$(SERVICE_NAME)/* /tmp/govendor/src
	mkdir -p /go/src/gitlab.wallstcn.com/$(CI_PROJECT_NAMESPACE)/
	cp -R "/builds/$(CI_PROJECT_NAMESPACE)/$(SERVICE_NAME)" "/go/src/gitlab.wallstcn.com/$(CI_PROJECT_NAMESPACE)/$(SERVICE_NAME)/"

test:
	go test
