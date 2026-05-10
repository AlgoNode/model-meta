GO       ?= go
PKG      := ./...
COVERAGE := coverage.txt

.PHONY: all build test test-race cover vet fmt tidy lint clean

all: fmt vet test

build:
	$(GO) build $(PKG)

test:
	$(GO) test -count=1 $(PKG)

test-race:
	$(GO) test -race -count=1 $(PKG)

cover:
	$(GO) test -count=1 -covermode=atomic -coverprofile=$(COVERAGE) $(PKG)
	$(GO) tool cover -func=$(COVERAGE) | tail -1

vet:
	$(GO) vet $(PKG)

fmt:
	$(GO) fmt $(PKG)

tidy:
	$(GO) mod tidy

clean:
	rm -f $(COVERAGE)
	rm -rf bin dist
