// Copyright (c) 2023 The TQUIC Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::connection::path::PathMap;
use crate::connection::space::PacketNumSpaceMap;
use crate::connection::stream::StreamMap;
use crate::multipath_scheduler::traffic_metrics::TrafficMetricsCollector;
use crate::multipath_scheduler::MultipathScheduler;
use crate::Error;
use crate::MultipathConfig;
use crate::Result;

/// RoundRobinScheduler distributes traffic equally across available paths.
///
/// On each selection it picks the sendable path that has sent the fewest
/// bytes so far, breaking ties by path id. This gives a closer equal split of
/// traffic even when paths have different packet sizes or capacities.
pub struct RoundRobinScheduler {
    metrics: TrafficMetricsCollector,
}

impl RoundRobinScheduler {
    pub fn new(_conf: &MultipathConfig) -> RoundRobinScheduler {
        RoundRobinScheduler { metrics: TrafficMetricsCollector::new() }
    }
}



impl MultipathScheduler for RoundRobinScheduler {
    /// Select the sendable path with the fewest bytes sent so far.
    fn on_select(
        &mut self,
        paths: &mut PathMap,
        spaces: &mut PacketNumSpaceMap,
        streams: &mut StreamMap,
    ) -> Result<usize> {
        let mut best: Option<(usize, u64)> = None;

        for (pid, path) in paths.iter_mut() {
            if !path.active() || !path.recovery.can_send() {
                continue;
            }
            let bytes = path.recovery.stats.sent_bytes;
            match best {
                None => best = Some((pid, bytes)),
                Some((best_pid, best_bytes)) => {
                    if bytes < best_bytes || (bytes == best_bytes && pid < best_pid) {
                        best = Some((pid, bytes));
                    }
                }
            }
        }

        match best {
            Some((pid, _bytes)) => {
                self.metrics.record(pid, paths);
                Ok(pid)
            }
            None => Err(Error::Done),
        }
    }

    fn scheduler_metrics_jsonl(&self) -> Vec<String> {
        self.metrics.metrics.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multipath_scheduler::tests::*;

    #[test]
    fn round_robin_single_available_path() -> Result<()> {
        let mut t = MultipathTester::new()?;

        let mut s = RoundRobinScheduler::new(&MultipathConfig::default());
        assert_eq!(s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?, 0);
        assert_eq!(s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?, 0);
        Ok(())
    }

    #[test]
    fn round_robin_multi_available_path() -> Result<()> {
        let mut t = MultipathTester::new()?;
        t.add_path("127.0.0.1:443", "127.0.0.2:8443", 50)?;
        t.add_path("127.0.0.1:443", "127.0.0.3:8443", 150)?;
        t.add_path("127.0.0.1:443", "127.0.0.4:8443", 100)?;

        let mut s = RoundRobinScheduler::new(&MultipathConfig::default());
        assert_eq!(s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?, 0);
        assert_eq!(s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?, 1);
        assert_eq!(s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?, 2);
        assert_eq!(s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?, 3);

        t.set_path_active(1, false)?;
        assert_eq!(s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?, 0);
        assert_eq!(s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?, 2);

        t.set_path_active(3, false)?;
        assert_eq!(s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?, 0);
        Ok(())
    }

    #[test]
    fn round_robin_prefers_lower_byte_usage() -> Result<()> {
        let mut t = MultipathTester::new()?;
        t.add_path("127.0.0.1:443", "127.0.0.2:8443", 50)?;

        {
            let path = t.paths.get_mut(0)?;
            path.recovery.stats.sent_bytes = 1200;
        }
        {
            let path = t.paths.get_mut(1)?;
            path.recovery.stats.sent_bytes = 600;
        }

        let mut s = RoundRobinScheduler::new(&MultipathConfig::default());
        assert_eq!(s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?, 1);
        Ok(())
    }

    #[test]
    fn round_robin_no_available_path() -> Result<()> {
        let mut t = MultipathTester::new()?;
        t.set_path_active(0, false)?;

        let mut s = RoundRobinScheduler::new(&MultipathConfig::default());
        assert_eq!(
            s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams),
            Err(Error::Done)
        );
        Ok(())
    }
}
