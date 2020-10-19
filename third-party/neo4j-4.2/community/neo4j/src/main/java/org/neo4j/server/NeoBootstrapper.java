/*
 * Copyright (c) 2002-2020 "Neo4j,"
 * Neo4j Sweden AB [http://neo4j.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.neo4j.server;

import sun.misc.Signal;

import java.io.Closeable;
import java.nio.file.Path;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.neo4j.configuration.Config;
import org.neo4j.configuration.GraphDatabaseInternalSettings;
import org.neo4j.configuration.GraphDatabaseSettings;
import org.neo4j.configuration.connectors.HttpConnector;
import org.neo4j.dbms.api.DatabaseManagementService;
import org.neo4j.graphdb.TransactionFailureException;
import org.neo4j.graphdb.facade.GraphDatabaseDependencies;
import org.neo4j.io.IOUtils;
import org.neo4j.kernel.internal.Version;
import org.neo4j.logging.Log;
import org.neo4j.logging.log4j.Log4jLogProvider;
import org.neo4j.logging.log4j.LogConfig;
import org.neo4j.logging.log4j.Neo4jLoggerContext;
import org.neo4j.server.logging.JULBridge;
import org.neo4j.server.logging.JettyLogBridge;
import org.neo4j.util.VisibleForTesting;

import static java.lang.String.format;

public abstract class NeoBootstrapper implements Bootstrapper
{
    public static final int OK = 0;
    private static final int WEB_SERVER_STARTUP_ERROR_CODE = 1;
    private static final int GRAPH_DATABASE_STARTUP_ERROR_CODE = 2;
    private static final String SIGTERM = "TERM";
    private static final String SIGINT = "INT";

    private volatile DatabaseManagementService databaseManagementService;
    private volatile Closeable userLogFileStream;
    private Thread shutdownHook;
    private GraphDatabaseDependencies dependencies = GraphDatabaseDependencies.newDependencies();
    private Log log;
    private String serverAddress = "unknown address";

    public static int start( Bootstrapper boot, String... argv )
    {
        CommandLineArgs args = CommandLineArgs.parse( argv );

        if ( args.version() )
        {
            System.out.println( "neo4j " + Version.getNeo4jVersion() );
            return 0;
        }

        if ( args.homeDir() == null )
        {
            throw new ServerStartupException( "Argument --home-dir is required and was not provided." );
        }

        return boot.start( args.homeDir(), args.configFile(), args.configOverrides() );
    }

    @VisibleForTesting
    public final int start( Path homeDir, Map<String, String> configOverrides )
    {
        return start( homeDir, null, configOverrides );
    }

    @Override
    public final int start( Path homeDir, Path configFile, Map<String, String> configOverrides )
    {
        addShutdownHook();
        installSignalHandlers();
        Config config = Config.newBuilder()
                .setDefaults( GraphDatabaseSettings.SERVER_DEFAULTS )
                .fromFileNoThrow( configFile )
                .setRaw( configOverrides )
                .set( GraphDatabaseSettings.neo4j_home, homeDir.toAbsolutePath() )
                .build();
        Log4jLogProvider userLogProvider = setupLogging( config );
        this.userLogFileStream = userLogProvider;

        dependencies = dependencies.userLogProvider( userLogProvider );
        log = userLogProvider.getLog( getClass() );
        config.setLogger( log );

        try
        {
            serverAddress = HttpConnector.listen_address.toString();

            log.info( "Starting..." );
            databaseManagementService = createNeo( config, dependencies );
            log.info( "Started." );

            return OK;
        }
        catch ( ServerStartupException e )
        {
            e.describeTo( log );
            return WEB_SERVER_STARTUP_ERROR_CODE;
        }
        catch ( TransactionFailureException tfe )
        {
            String locationMsg = (databaseManagementService == null) ? "" :
                                 " Another process may be using databases at location: " + config.get( GraphDatabaseInternalSettings.databases_root_path );
            log.error( format( "Failed to start Neo4j on %s.", serverAddress ) + locationMsg, tfe );
            return GRAPH_DATABASE_STARTUP_ERROR_CODE;
        }
        catch ( Exception e )
        {
            log.error( format( "Failed to start Neo4j on %s.", serverAddress ), e );
            return WEB_SERVER_STARTUP_ERROR_CODE;
        }
    }

    @Override
    public int stop()
    {
        String location = "unknown location";
        try
        {
            doShutdown();

            removeShutdownHook();

            return 0;
        }
        catch ( Exception e )
        {
            log.error( "Failed to cleanly shutdown Neo Server on port [%s], database [%s]. Reason [%s] ",
                    serverAddress, location, e.getMessage(), e );
            return 1;
        }
    }

    public boolean isRunning()
    {
        return databaseManagementService != null;
    }

    public DatabaseManagementService getDatabaseManagementService()
    {
        return databaseManagementService;
    }

    public Log getLog()
    {
        return log;
    }

    protected abstract DatabaseManagementService createNeo( Config config, GraphDatabaseDependencies dependencies );

    private Log4jLogProvider setupLogging( Config config )
    {

        LogConfig.Builder builder =
                LogConfig.createBuilder( config.get( GraphDatabaseSettings.store_user_log_path ), config.get( GraphDatabaseSettings.store_internal_log_level ) )
                        .withTimezone( config.get( GraphDatabaseSettings.db_timezone ) )
                        .withFormat( config.get( GraphDatabaseInternalSettings.log_format ) )
                        .withCategory( false )
                        .withRotation( config.get( GraphDatabaseSettings.store_user_log_rotation_threshold ),
                                config.get( GraphDatabaseSettings.store_user_log_max_archives ) );

        if ( config.get( GraphDatabaseSettings.store_user_log_to_stdout ) )
        {
            builder.logToSystemOut();
        }

        Neo4jLoggerContext ctx = builder.build();
        Log4jLogProvider userLogProvider = new Log4jLogProvider( ctx );

        JULBridge.resetJUL();
        Logger.getLogger( "" ).setLevel( Level.WARNING );
        JULBridge.forwardTo( userLogProvider );
        JettyLogBridge.setLogProvider( userLogProvider );
        return userLogProvider;
    }

    // Exit gracefully if possible
    private static void installSignalHandlers()
    {
        installSignalHandler( SIGTERM, false ); // SIGTERM is invoked when system service is stopped
        installSignalHandler( SIGINT, true ); // SIGINT is invoked when user hits ctrl-c  when running `neo4j console`
    }

    private static void installSignalHandler( String sig, boolean tolerateErrors )
    {
        try
        {
            // System.exit() will trigger the shutdown hook
            Signal.handle( new Signal( sig ), signal -> System.exit( 0 ) );
        }
        catch ( Throwable e )
        {
            if ( !tolerateErrors )
            {
                throw e;
            }
            // Errors occur on IBM JDK with IllegalArgumentException: Signal already used by VM: INT
            // I can't find anywhere where we send a SIGINT to neo4j process so I don't think this is that important
        }
    }

    private void doShutdown()
    {
        if ( databaseManagementService != null )
        {
            log.info( "Stopping..." );
            databaseManagementService.shutdown();
            log.info( "Stopped." );
        }
        if ( userLogFileStream != null )
        {
            closeUserLogFileStream();
        }
    }

    private void closeUserLogFileStream()
    {
        IOUtils.closeAllUnchecked( userLogFileStream );
    }

    private void addShutdownHook()
    {
        shutdownHook = new Thread( () -> {
            log.info( "Neo4j Server shutdown initiated by request" );
            doShutdown();
        } );
        Runtime.getRuntime().addShutdownHook( shutdownHook );
    }

    private void removeShutdownHook()
    {
        if ( shutdownHook != null )
        {
            if ( !Runtime.getRuntime().removeShutdownHook( shutdownHook ) )
            {
                log.warn( "Unable to remove shutdown hook" );
            }
        }
    }
}
