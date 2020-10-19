echo "export CADVISOR_URL=$CADVISOR_URL" > /cron/cronfile
echo "export COLLECTOR_URL=$COLLECTOR_URL" >> /cron/cronfile
mkdir /cron/bin/Cron
ln /cron/bin/runcron /cron/bin/Cron/sender.sh
rsyslogd && cron && tail -f /var/log/syslog /var/log/cron.log
