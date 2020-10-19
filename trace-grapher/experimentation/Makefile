

.PHONY: \
	prepare-common \
	bookinfo-start \
	bookinfo-stop


prepare-common:
	@-kubectl label namespace default istio-injection=enabled


bookinfo-start: prepare-common
	@kubectl apply -k ./with-bookinfo/


bookinfo-stop:
	@kubectl delete -k ./with-bookinfo/


emojivoto-start: prepare-common
	@kubectl apply -k ./with-emojivoto/


emojivoto-stop:
	@kubectl delete -k ./with-emojivoto/