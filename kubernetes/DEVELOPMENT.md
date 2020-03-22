## Development Guidance

This project uses go modules. We suggest to use golang 1.13.x for development and make sure`GO111MODULE` is enabled.

### Setup development environment

Fork [PyTorch/Elastic](https://github.com/pytorch/elastic)

```shell
mkdir -p ${GOPATH}/src/github.com/pytorch/
cd ${GOPATH}/src/github.com/pytorch
git clone git@github.com:${GITHUB_USER}/elastic.git

# operator codes is under kubernetes directory
cd elastic/kubernetes
```

### Download dependencies.

```shell
go mod download
```

### Build the binary locally

```shell
make manager
```

### Test Binaries locally

```shell
./bin/manager
```

### Run Tests

```shell
go test ./... -coverprofile cover.out
```

### Build container image

```shell
# It requires you to build binary locally first.
docker build -t ${your_dockerhub_username}/torch-elastic-operator:latest .
```
