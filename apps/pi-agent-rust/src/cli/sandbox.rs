use clap::ValueEnum;
use pi_agent_runtime::{DeltaBoxFsMode, SandboxMode};
use pi_agent_rust::deltabox::DeltaBoxDisposition;

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum SandboxModeArg {
    Managed,
    Disabled,
    Deltabox,
}

impl From<SandboxModeArg> for SandboxMode {
    fn from(value: SandboxModeArg) -> Self {
        match value {
            SandboxModeArg::Managed => Self::Managed,
            SandboxModeArg::Disabled => Self::Disabled,
            SandboxModeArg::Deltabox => Self::DeltaBox,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum DeltaBoxFsModeArg {
    CbcowThrough,
    CbcowPrivate,
    CopyPrivate,
}

impl From<DeltaBoxFsModeArg> for DeltaBoxFsMode {
    fn from(value: DeltaBoxFsModeArg) -> Self {
        match value {
            DeltaBoxFsModeArg::CbcowThrough => Self::CbCowThrough,
            DeltaBoxFsModeArg::CbcowPrivate => Self::CbCowPrivate,
            DeltaBoxFsModeArg::CopyPrivate => Self::CopyPrivate,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum DeltaBoxActionArg {
    Status,
    Diff,
    Commit,
    Abort,
    Stop,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum DeltaBoxSuccessArg {
    Review,
    Commit,
    Abort,
}

impl From<DeltaBoxSuccessArg> for DeltaBoxDisposition {
    fn from(value: DeltaBoxSuccessArg) -> Self {
        match value {
            DeltaBoxSuccessArg::Review => Self::Review,
            DeltaBoxSuccessArg::Commit => Self::Commit,
            DeltaBoxSuccessArg::Abort => Self::Abort,
        }
    }
}
