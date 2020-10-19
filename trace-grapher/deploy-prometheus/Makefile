OVERLAY ?= docker-desktop

.PHONY: \
	install-prometheus \
	remove-prometheus

install-prometheus:
	@kubectl apply -k ./overlays/$(OVERLAY)

remove-prometheus:
	@-kubectl delete -k ./overlays/$(OVERLAY)