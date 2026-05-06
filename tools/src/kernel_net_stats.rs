// Kernel-level network drop counters read directly from the Linux virtual
// filesystems (/proc, /sys).  These are the *same* values that an eBPF TC
// hook would observe, maintained by the kernel for zero overhead.
//
// Three independent drop sources are tracked:
//
// 1. UDP ingress drops — datagrams discarded because the receiving socket's
//    SO_RCVBUF was full (kernel counter: UdpRcvbufErrors in /proc/net/snmp).
//    The app-level QUIC stack never sees these packets at all.
//
// 2. NIC / driver RX drops — frames dropped by the NIC driver before they
//    reach the socket layer (/sys/class/net/<iface>/statistics/rx_dropped).
//    Causes: NIC ring-buffer overflow, driver budget exhaustion.
//
// 3. Egress qdisc TX drops — frames dropped by the traffic-control qdisc
//    after the kernel accepted the sendmsg() call
//    (/sys/class/net/<iface>/statistics/tx_dropped).
//    The app believes the packet was sent; the kernel discarded it instead.
//
// An optional bonus counter:
// 4. rx_missed_errors — hardware-level NIC ring overflow
//    (/sys/class/net/<iface>/statistics/rx_missed_errors).

use std::fs;
use std::io;

/// A snapshot of kernel-level network drop counters at a single point in time.
#[derive(Debug, Default, Clone, Copy)]
pub struct KernelNetSnapshot {
    /// Global UDP datagrams discarded because the socket receive buffer was
    /// full.  Source: /proc/net/snmp  Udp: RcvbufErrors
    pub udp_rcvbuf_drops: u64,

    /// NIC / driver level ingress drops (ring-buffer overflow, budget etc.).
    /// Source: /sys/class/net/<iface>/statistics/rx_dropped
    pub nic_rx_dropped: u64,

    /// Traffic-control qdisc egress drops.
    /// Source: /sys/class/net/<iface>/statistics/tx_dropped
    pub qdisc_tx_dropped: u64,

    /// NIC hardware RX ring misses (reported by the NIC itself).
    /// Source: /sys/class/net/<iface>/statistics/rx_missed_errors
    pub nic_rx_missed: u64,
}

impl KernelNetSnapshot {
    /// Delta between two snapshots (self is newer, `prev` is older).
    /// Saturating subtraction handles counter wraps / resets gracefully.
    pub fn delta(&self, prev: &KernelNetSnapshot) -> KernelNetSnapshot {
        KernelNetSnapshot {
            udp_rcvbuf_drops: self.udp_rcvbuf_drops.saturating_sub(prev.udp_rcvbuf_drops),
            nic_rx_dropped:   self.nic_rx_dropped.saturating_sub(prev.nic_rx_dropped),
            qdisc_tx_dropped: self.qdisc_tx_dropped.saturating_sub(prev.qdisc_tx_dropped),
            nic_rx_missed:    self.nic_rx_missed.saturating_sub(prev.nic_rx_missed),
        }
    }

    /// Returns true when all counters are zero (nothing to display).
    pub fn all_zero(&self) -> bool {
        self.udp_rcvbuf_drops == 0
            && self.nic_rx_dropped == 0
            && self.qdisc_tx_dropped == 0
            && self.nic_rx_missed == 0
    }
}

/// Reads a single u64 from a one-line sysfs file.
fn read_sysfs_u64(path: &str) -> io::Result<u64> {
    let raw = fs::read_to_string(path)?;
    raw.trim()
        .parse::<u64>()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Parse /proc/net/snmp and return a specific field value.
///
/// The file is structured as alternating header / value rows for each
/// protocol section.  Example:
///
/// ```
/// Udp: InDatagrams NoPorts InErrors InCsumErrors IgnoredMulti ...
/// Udp: 123456      0       0        0             0            ...
/// ```
///
/// `section` is the prefix like "Udp:", `field` is the column name.
fn parse_proc_net_snmp(section: &str, field: &str) -> io::Result<u64> {
    let content = fs::read_to_string("/proc/net/snmp")?;
    let mut header_cols: Option<Vec<&str>> = None;

    for line in content.lines() {
        if !line.starts_with(section) {
            header_cols = None;
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        match &header_cols {
            None => {
                // This is the header row — remember column names.
                header_cols = Some(cols);
            }
            Some(hdr) => {
                // This is the value row.
                if let Some(idx) = hdr.iter().position(|&h| h == field) {
                    let val = cols.get(idx).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("field {field} missing in value row"),
                        )
                    })?;
                    return val
                        .parse::<u64>()
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e));
                }
                // Field not found in this section's header — reset and keep looking.
                header_cols = None;
            }
        }
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("section={section} field={field} not found in /proc/net/snmp"),
    ))
}

/// Capture a full snapshot.  `iface` may be `None` to skip per-interface
/// counters (NIC/qdisc drops will be zero in that case).
pub fn snapshot(iface: Option<&str>) -> KernelNetSnapshot {
    let udp_rcvbuf_drops = parse_proc_net_snmp("Udp:", "RcvbufErrors").unwrap_or(0);

    let (nic_rx_dropped, qdisc_tx_dropped, nic_rx_missed) = if let Some(dev) = iface {
        let base = format!("/sys/class/net/{dev}/statistics");
        let rx_dropped   = read_sysfs_u64(&format!("{base}/rx_dropped")).unwrap_or(0);
        let tx_dropped   = read_sysfs_u64(&format!("{base}/tx_dropped")).unwrap_or(0);
        let rx_missed    = read_sysfs_u64(&format!("{base}/rx_missed_errors")).unwrap_or(0);
        (rx_dropped, tx_dropped, rx_missed)
    } else {
        (0, 0, 0)
    };

    KernelNetSnapshot {
        udp_rcvbuf_drops,
        nic_rx_dropped,
        qdisc_tx_dropped,
        nic_rx_missed,
    }
}

/// Attempt to discover the default route interface from /proc/net/route.
/// Returns the name of the first interface with a default gateway (Destination == 0).
/// Falls back to `None` if the file cannot be parsed.
pub fn default_iface() -> Option<String> {
    let content = fs::read_to_string("/proc/net/route").ok()?;
    for line in content.lines().skip(1) {
        let cols: Vec<&str> = line.split_whitespace().collect();
        // col 0 = iface, col 1 = Destination (hex), col 2 = Gateway (hex)
        if cols.len() >= 3 && cols[1] == "00000000" {
            return Some(cols[0].to_string());
        }
    }
    None
}
