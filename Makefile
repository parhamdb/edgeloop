.PHONY: build test release clean cross-all cross-arm64 cross-armv7 cross-mips cross-x86

# Development
build:
	cargo build

test:
	cargo test --bin edgeloop

test-integration:
	cargo test --test integration_test

test-bench:
	cargo test --test benchmark -- --nocapture

test-all: test test-integration test-bench

# Release builds
release:
	cargo build --release

release-full:
	cargo build --release --features full

release-minimal:
	cargo build --release --no-default-features --features "llama-server,cli-transport"

# Cross-compilation (requires: cargo install cross)
cross-arm64:
	cross build --release --features full --target aarch64-unknown-linux-musl

cross-armv7:
	cross build --release --features full --target armv7-unknown-linux-musleabihf

cross-mips:
	cross build --release --no-default-features --features "llama-server,cli-transport" --target mips-unknown-linux-musl

cross-mipsel:
	cross build --release --no-default-features --features "llama-server,cli-transport" --target mipsel-unknown-linux-musl

cross-x86:
	cross build --release --features full --target x86_64-unknown-linux-musl

cross-all: cross-arm64 cross-armv7 cross-mips cross-x86

# Collect binaries
dist: cross-all
	mkdir -p dist
	cp target/aarch64-unknown-linux-musl/release/edgeloop dist/edgeloop-aarch64
	cp target/armv7-unknown-linux-musleabihf/release/edgeloop dist/edgeloop-armv7
	cp target/mips-unknown-linux-musl/release/edgeloop dist/edgeloop-mips
	cp target/x86_64-unknown-linux-musl/release/edgeloop dist/edgeloop-x86_64
	ls -lh dist/

clean:
	cargo clean
	rm -rf dist/
