include vars.mk

OVERLAY ?= docker-desktop

.PHONY: \
	prepare-trace-grapher-namespace \
	install-components \
	remove-components \
	purge


prepare-trace-grapher-namespace:
	@echo "Eventually creating Namespace: $(APPLICATION_NS)"
	@-kubectl create namespace "$(APPLICATION_NS)"


install-components: prepare-trace-grapher-namespace
	@kubectl apply -k ./overlays/$(OVERLAY)


remove-components:
	@-kubectl delete -k ./overlays/$(OVERLAY)


purge: remove-components
	@-kubectl delete namespace "$(APPLICATION_NS)"