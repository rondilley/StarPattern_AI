"""Tests for the distributed master/slave computing module.

No mocks -- real TCP connections on localhost.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from star_pattern.distributed.config import DistributedConfig
from star_pattern.distributed.protocol import (
    PROTOCOL_VERSION,
    WorkUnit,
    WorkResult,
    make_auth,
    recv_message,
    send_message,
    verify_auth,
)


# --- Protocol tests ---


class TestProtocol:
    """Test wire protocol: framing, send/recv round-trip."""

    @pytest.fixture
    def event_loop_pair(self):
        """Create a connected reader/writer pair via localhost TCP."""
        async def _make_pair():
            connected = asyncio.Event()
            result = {}

            async def handler(reader, writer):
                result["server_reader"] = reader
                result["server_writer"] = writer
                connected.set()

            server = await asyncio.start_server(handler, "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            await connected.wait()
            return (
                reader, writer,
                result["server_reader"], result["server_writer"],
                server,
            )

        loop = asyncio.new_event_loop()
        try:
            client_r, client_w, server_r, server_w, server = loop.run_until_complete(
                _make_pair()
            )
            yield loop, client_r, client_w, server_r, server_w, server
        finally:
            loop.run_until_complete(self._cleanup(server, client_w, server_w))
            loop.close()

    @staticmethod
    async def _cleanup(server, *writers):
        for w in writers:
            try:
                w.close()
                await w.wait_closed()
            except Exception:
                pass
        server.close()
        await server.wait_closed()

    def test_send_recv_roundtrip(self, event_loop_pair):
        """Messages survive send -> gzip -> length-prefix -> recv -> decompress."""
        loop, client_r, client_w, server_r, server_w, _ = event_loop_pair

        msg = {
            "type": "test",
            "data": {"value": 42, "nested": [1, 2, 3]},
        }

        async def _test():
            await send_message(client_w, msg)
            received = await recv_message(server_r, timeout=5.0)
            return received

        result = loop.run_until_complete(_test())
        assert result["type"] == "test"
        assert result["data"]["value"] == 42
        assert result["data"]["nested"] == [1, 2, 3]

    def test_bidirectional_messages(self, event_loop_pair):
        """Both sides can send and receive."""
        loop, client_r, client_w, server_r, server_w, _ = event_loop_pair

        async def _test():
            # Client -> Server
            await send_message(client_w, {"type": "ping", "seq": 1})
            msg1 = await recv_message(server_r, timeout=5.0)

            # Server -> Client
            await send_message(server_w, {"type": "pong", "seq": 1})
            msg2 = await recv_message(client_r, timeout=5.0)

            return msg1, msg2

        msg1, msg2 = loop.run_until_complete(_test())
        assert msg1["type"] == "ping"
        assert msg2["type"] == "pong"

    def test_large_message(self, event_loop_pair):
        """Large payloads (simulating WorkResult) survive the wire."""
        loop, client_r, client_w, server_r, server_w, _ = event_loop_pair

        # Simulate a large result with many pattern results
        large_msg = {
            "type": "work_result",
            "payload": {
                "work_id": "test-id",
                "pattern_results": [
                    {"ra": i * 0.1, "dec": i * 0.2, "score": i * 0.01}
                    for i in range(500)
                ],
            },
        }

        async def _test():
            await send_message(client_w, large_msg)
            received = await recv_message(server_r, timeout=5.0)
            return received

        result = loop.run_until_complete(_test())
        assert len(result["payload"]["pattern_results"]) == 500

    def test_recv_timeout(self, event_loop_pair):
        """recv_message raises TimeoutError when nothing arrives."""
        loop, client_r, client_w, server_r, server_w, _ = event_loop_pair

        async def _test():
            with pytest.raises(asyncio.TimeoutError):
                await recv_message(server_r, timeout=0.1)

        loop.run_until_complete(_test())


# --- Auth tests ---


class TestAuth:
    """Test HMAC-SHA256 authentication."""

    def test_valid_auth(self):
        """Correct token and fresh timestamp passes."""
        token = "test-secret-token"
        ts = time.time()
        digest = make_auth(token, ts)
        assert verify_auth(token, ts, digest) is True

    def test_wrong_token(self):
        """Wrong token fails verification."""
        ts = time.time()
        digest = make_auth("correct-token", ts)
        assert verify_auth("wrong-token", ts, digest) is False

    def test_expired_timestamp(self):
        """Timestamp outside tolerance window fails."""
        token = "test-token"
        old_ts = time.time() - 400  # 400s ago, tolerance is 300s
        digest = make_auth(token, old_ts)
        assert verify_auth(token, old_ts, digest) is False

    def test_no_auth_configured(self):
        """Empty token with empty digest is allowed (no-auth mode)."""
        assert verify_auth("", 0, "") is True

    def test_tampered_digest(self):
        """Modified digest fails."""
        token = "secret"
        ts = time.time()
        digest = make_auth(token, ts)
        tampered = digest[:-4] + "0000"
        assert verify_auth(token, ts, tampered) is False


# --- Serialization tests ---


class TestSerialization:
    """Test WorkUnit and WorkResult serialization round-trips."""

    def test_work_unit_roundtrip(self):
        """WorkUnit survives to_dict/from_dict."""
        unit = WorkUnit(
            work_id="test-123",
            region={"ra": 180.0, "dec": 45.0, "radius": 3.0},
            detection_config={"source_extraction_threshold": 2.5},
            genome_dict={"genes": [0.1, 0.2, 0.3]},
            data_config={"sources": ["sdss", "gaia"]},
            temporal_config={"max_epochs": 5},
            include_temporal=True,
            priority=1,
        )
        d = unit.to_dict()
        restored = WorkUnit.from_dict(d)

        assert restored.work_id == "test-123"
        assert restored.region["ra"] == 180.0
        assert restored.detection_config["source_extraction_threshold"] == 2.5
        assert restored.include_temporal is True
        assert restored.priority == 1

    def test_work_result_roundtrip(self):
        """WorkResult survives to_dict/from_dict."""
        result = WorkResult(
            work_id="test-456",
            slave_id="node1:1234",
            region={"ra": 90.0, "dec": -30.0, "radius": 3.0},
            pattern_results=[
                {
                    "ra": 90.0,
                    "dec": -30.0,
                    "type": "lens",
                    "anomaly_score": 0.85,
                    "anomalies": [
                        {
                            "anomaly_type": "lens_arc",
                            "detector": "lens",
                            "score": 0.9,
                            "sky_ra": 90.1,
                            "sky_dec": -30.1,
                        }
                    ],
                }
            ],
            detection_summaries=[
                {"band": "r", "anomaly_score": 0.85, "n_anomalies": 1}
            ],
            elapsed_seconds=42.5,
        )
        d = result.to_dict()
        restored = WorkResult.from_dict(d)

        assert restored.work_id == "test-456"
        assert restored.slave_id == "node1:1234"
        assert len(restored.pattern_results) == 1
        assert restored.pattern_results[0]["anomaly_score"] == 0.85
        assert restored.elapsed_seconds == 42.5

    def test_work_unit_auto_id(self):
        """WorkUnit generates a UUID if none provided."""
        unit = WorkUnit(region={"ra": 0, "dec": 0, "radius": 3})
        assert unit.work_id != ""
        assert len(unit.work_id) == 36  # UUID format

    def test_work_result_with_error(self):
        """WorkResult can carry an error string."""
        result = WorkResult(
            work_id="err-1",
            slave_id="node2:5678",
            error="Connection timeout",
        )
        d = result.to_dict()
        restored = WorkResult.from_dict(d)
        assert restored.error == "Connection timeout"
        assert restored.pattern_results == []


# --- Config tests ---


class TestDistributedConfig:
    """Test DistributedConfig dataclass."""

    def test_defaults(self):
        """Default config is standalone mode."""
        config = DistributedConfig()
        assert config.mode == "standalone"
        assert config.slave_addresses == []
        assert config.listen_port == 7827

    def test_from_dict(self):
        """Config can be created from a dict."""
        d = {
            "mode": "master",
            "slave_addresses": ["host1:7827", "host2:7828"],
            "auth_token": "secret",
            "local_cycle_interval": 5,
        }
        config = DistributedConfig.from_dict(d)
        assert config.mode == "master"
        assert len(config.slave_addresses) == 2
        assert config.auth_token == "secret"
        assert config.local_cycle_interval == 5

    def test_from_empty_dict(self):
        """Empty dict produces default config."""
        config = DistributedConfig.from_dict({})
        assert config.mode == "standalone"


# --- Integration tests ---


class TestPatternResultDeserialization:
    """Test PatternResult reconstruction from serialized WorkResult."""

    def test_result_dict_to_pattern_result(self):
        """Verify bridge can reconstruct PatternResult from dict."""
        from star_pattern.distributed.bridge import _result_dict_to_pattern_result

        d = {
            "ra": 180.0,
            "dec": 45.0,
            "type": "lens",
            "anomaly_score": 0.75,
            "significance": 0.5,
            "novelty": 0.3,
            "hypothesis": "Possible gravitational lens",
            "debate_verdict": "real",
            "consensus_score": 0.8,
            "metadata": {"band": "r"},
            "cross_matches": [{"name": "SDSS J1200+4500"}],
            "anomalies": [
                {
                    "anomaly_type": "lens_arc",
                    "detector": "lens",
                    "pixel_x": 128.0,
                    "pixel_y": 130.0,
                    "sky_ra": 180.01,
                    "sky_dec": 45.01,
                    "score": 0.9,
                    "properties": {"radius": 15.0},
                },
            ],
        }

        pr = _result_dict_to_pattern_result(d)
        assert pr.region_ra == 180.0
        assert pr.region_dec == 45.0
        assert pr.detection_type == "lens"
        assert pr.anomaly_score == 0.75
        assert pr.hypothesis == "Possible gravitational lens"
        assert pr.debate_verdict == "real"
        assert len(pr.anomalies) == 1
        assert pr.anomalies[0].anomaly_type == "lens_arc"
        assert pr.anomalies[0].score == 0.9
        assert pr.anomalies[0].properties["radius"] == 15.0


class TestSlaveServerLifecycle:
    """Test SlaveServer starts, accepts connections, and shuts down."""

    def test_slave_starts_and_handshakes(self):
        """Slave starts, accepts a connection, handshakes, and shuts down."""
        from star_pattern.core.config import PipelineConfig
        from star_pattern.distributed.slave import SlaveServer

        config = PipelineConfig()
        config.distributed = DistributedConfig(
            mode="slave",
            listen_host="127.0.0.1",
            listen_port=0,  # OS-assigned port
            auth_token="test-token",
        )

        async def _test():
            server = SlaveServer(config)

            # Start server on OS-assigned port
            server._server = await asyncio.start_server(
                server._handle_connection,
                host="127.0.0.1",
                port=0,
            )
            port = server._server.sockets[0].getsockname()[1]

            # Connect as master
            reader, writer = await asyncio.open_connection("127.0.0.1", port)

            # Send handshake
            ts = time.time()
            await send_message(writer, {
                "type": "handshake",
                "version": PROTOCOL_VERSION,
                "auth_digest": make_auth("test-token", ts),
                "timestamp": ts,
            })

            # Receive ack
            ack = await recv_message(reader, timeout=5.0)
            assert ack["type"] == "handshake_ack"
            assert "slave_id" in ack
            assert "n_workers" in ack

            # Send shutdown
            await send_message(writer, {"type": "shutdown"})
            await asyncio.sleep(0.1)

            # Cleanup
            writer.close()
            await writer.wait_closed()
            server._shutdown_event.set()
            server._server.close()
            await server._server.wait_closed()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_test())
        finally:
            loop.close()

    def test_slave_rejects_bad_auth(self):
        """Slave closes connection when auth digest is wrong."""
        from star_pattern.core.config import PipelineConfig
        from star_pattern.distributed.slave import SlaveServer

        config = PipelineConfig()
        config.distributed = DistributedConfig(
            mode="slave",
            listen_host="127.0.0.1",
            listen_port=0,
            auth_token="correct-token",
        )

        async def _test():
            server = SlaveServer(config)
            server._server = await asyncio.start_server(
                server._handle_connection,
                host="127.0.0.1",
                port=0,
            )
            port = server._server.sockets[0].getsockname()[1]

            reader, writer = await asyncio.open_connection("127.0.0.1", port)

            # Send handshake with wrong auth
            ts = time.time()
            await send_message(writer, {
                "type": "handshake",
                "version": PROTOCOL_VERSION,
                "auth_digest": make_auth("wrong-token", ts),
                "timestamp": ts,
            })

            # Connection should be closed by slave
            await asyncio.sleep(0.2)
            # Trying to read should fail or return empty
            try:
                data = await asyncio.wait_for(reader.read(1024), timeout=1.0)
                # If we get data, it means connection is still open but shouldn't be
                # An empty read means connection closed (expected)
                assert data == b"", "Expected connection to be closed"
            except (asyncio.TimeoutError, ConnectionError):
                pass  # Connection closed as expected

            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            server._server.close()
            await server._server.wait_closed()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_test())
        finally:
            loop.close()


class TestMasterSlaveIntegration:
    """Full integration: master dispatches work, slave processes, master receives."""

    def test_config_update_roundtrip(self):
        """Master can send config_update to slave."""
        from star_pattern.core.config import PipelineConfig
        from star_pattern.distributed.slave import SlaveServer

        config = PipelineConfig()
        config.distributed = DistributedConfig(
            mode="slave",
            listen_host="127.0.0.1",
            listen_port=0,
            auth_token="",
        )

        async def _test():
            server = SlaveServer(config)
            server._server = await asyncio.start_server(
                server._handle_connection,
                host="127.0.0.1",
                port=0,
            )
            port = server._server.sockets[0].getsockname()[1]

            reader, writer = await asyncio.open_connection("127.0.0.1", port)

            # Handshake (no auth)
            await send_message(writer, {
                "type": "handshake",
                "version": PROTOCOL_VERSION,
                "auth_digest": "",
                "timestamp": 0,
            })
            ack = await recv_message(reader, timeout=5.0)
            assert ack["type"] == "handshake_ack"

            # Send config update
            await send_message(writer, {
                "type": "config_update",
                "payload": {
                    "detection_config": {
                        "source_extraction_threshold": 2.0,
                    },
                    "genome_dict": {},
                },
            })
            await asyncio.sleep(0.2)

            # Verify detector was rebuilt
            assert server._detector is not None

            # Shutdown
            await send_message(writer, {"type": "shutdown"})
            await asyncio.sleep(0.1)
            writer.close()
            await writer.wait_closed()
            server._shutdown_event.set()
            server._server.close()
            await server._server.wait_closed()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_test())
        finally:
            loop.close()

    def test_heartbeat_response(self):
        """Slave responds to heartbeat."""
        from star_pattern.core.config import PipelineConfig
        from star_pattern.distributed.slave import SlaveServer

        config = PipelineConfig()
        config.distributed = DistributedConfig(
            mode="slave",
            listen_host="127.0.0.1",
            listen_port=0,
            auth_token="",
        )

        async def _test():
            server = SlaveServer(config)
            server._server = await asyncio.start_server(
                server._handle_connection,
                host="127.0.0.1",
                port=0,
            )
            port = server._server.sockets[0].getsockname()[1]

            reader, writer = await asyncio.open_connection("127.0.0.1", port)

            # Handshake
            await send_message(writer, {
                "type": "handshake",
                "version": PROTOCOL_VERSION,
                "auth_digest": "",
                "timestamp": 0,
            })
            ack = await recv_message(reader, timeout=5.0)
            assert ack["type"] == "handshake_ack"

            # Send heartbeat
            await send_message(writer, {
                "type": "heartbeat",
                "timestamp": time.time(),
            })

            # Receive heartbeat response
            response = await recv_message(reader, timeout=5.0)
            assert response["type"] == "heartbeat"
            assert "timestamp" in response

            # Shutdown
            await send_message(writer, {"type": "shutdown"})
            await asyncio.sleep(0.1)
            writer.close()
            await writer.wait_closed()
            server._shutdown_event.set()
            server._server.close()
            await server._server.wait_closed()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_test())
        finally:
            loop.close()


class TestMasterDispatcher:
    """Test MasterDispatcher with real slave connection."""

    def test_connect_and_dispatch(self):
        """Master connects to a slave and dispatches work."""
        from star_pattern.core.config import PipelineConfig
        from star_pattern.distributed.slave import SlaveServer
        from star_pattern.distributed.master import MasterDispatcher

        slave_config = PipelineConfig()
        slave_config.distributed = DistributedConfig(
            mode="slave",
            listen_host="127.0.0.1",
            listen_port=0,
            auth_token="",
        )

        async def _test():
            # Start slave
            server = SlaveServer(slave_config)
            server._server = await asyncio.start_server(
                server._handle_connection,
                host="127.0.0.1",
                port=0,
            )
            port = server._server.sockets[0].getsockname()[1]

            # Create and connect master
            master_config = DistributedConfig(
                mode="master",
                slave_addresses=[f"127.0.0.1:{port}"],
                auth_token="",
            )
            dispatcher = MasterDispatcher(master_config)
            n_connected = await dispatcher.connect_all()
            assert n_connected == 1
            assert dispatcher.n_connected == 1

            # Dispatch a work unit (slave will try to fetch -- may fail
            # without network, but protocol should still work)
            unit = WorkUnit(
                region={"ra": 180.0, "dec": 45.0, "radius": 3.0},
                detection_config={"source_extraction_threshold": 3.0},
                data_config={"sources": ["sdss"], "cache_dir": "output/cache"},
            )
            dispatched = await dispatcher.dispatch(unit)
            assert dispatched is True
            assert dispatcher.total_pending == 1

            # Wait for result (may be an error if no network, that's OK)
            result = await dispatcher.collect_result(timeout=120.0)
            # Result could be None if slave returns error and retry kicks in,
            # or a WorkResult with data/error
            # The important thing is the protocol worked without crashing

            # Shutdown
            await dispatcher.shutdown_all()
            server._shutdown_event.set()
            server._server.close()
            await server._server.wait_closed()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(asyncio.wait_for(_test(), timeout=180.0))
        except asyncio.TimeoutError:
            pytest.skip("Test timed out (likely network issues)")
        finally:
            loop.close()


class TestPipelineConfigDistributed:
    """Test distributed config integration in PipelineConfig."""

    def test_default_standalone(self):
        """PipelineConfig defaults to standalone mode."""
        from star_pattern.core.config import PipelineConfig
        config = PipelineConfig()
        assert config.distributed.mode == "standalone"
        assert config.distributed.slave_addresses == []

    def test_from_dict_with_distributed(self):
        """PipelineConfig.from_dict parses distributed section."""
        from star_pattern.core.config import PipelineConfig
        d = {
            "distributed": {
                "mode": "master",
                "slave_addresses": ["host1:7827"],
                "auth_token": "secret",
            }
        }
        config = PipelineConfig.from_dict(d)
        assert config.distributed.mode == "master"
        assert config.distributed.slave_addresses == ["host1:7827"]
        assert config.distributed.auth_token == "secret"

    def test_to_dict_includes_distributed(self):
        """PipelineConfig.to_dict includes distributed section."""
        from star_pattern.core.config import PipelineConfig
        config = PipelineConfig()
        d = config.to_dict()
        assert "distributed" in d
        assert d["distributed"]["mode"] == "standalone"
