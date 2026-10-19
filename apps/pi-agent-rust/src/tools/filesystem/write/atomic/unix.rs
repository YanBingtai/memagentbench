//! Descriptor-relative atomic replacement for Unix platforms.

use std::{
    ffi::OsStr,
    fs::File,
    io::{self, Write},
    path::{Component, Path},
};

use rustix::fs::{openat, renameat, unlinkat, AtFlags, Mode, OFlags};

pub(super) fn write(
    workspace: &Path,
    parent: &Path,
    temporary_name: &OsStr,
    target_name: &OsStr,
    content: &[u8],
) -> io::Result<()> {
    let relative_parent = parent.strip_prefix(workspace).map_err(|_| {
        io::Error::new(
            io::ErrorKind::PermissionDenied,
            "write parent escaped workspace",
        )
    })?;
    let directory = open_parent(workspace, relative_parent)?;
    let temporary = openat(
        &directory,
        temporary_name,
        OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::from_raw_mode(0o600),
    )?;
    let mut temporary = File::from(temporary);
    if let Err(error) = temporary
        .write_all(content)
        .and_then(|_| temporary.sync_all())
    {
        let _ = unlinkat(&directory, temporary_name, AtFlags::empty());
        return Err(error);
    }
    drop(temporary);
    if let Err(error) = renameat(&directory, temporary_name, &directory, target_name) {
        let _ = unlinkat(&directory, temporary_name, AtFlags::empty());
        return Err(error.into());
    }
    Ok(())
}

fn open_parent(workspace: &Path, relative_parent: &Path) -> io::Result<File> {
    let mut directory = File::open(workspace)?;
    for component in relative_parent.components() {
        let Component::Normal(name) = component else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "write parent contains an invalid path component",
            ));
        };
        let next = openat(
            &directory,
            name,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )?;
        directory = File::from(next);
    }
    Ok(directory)
}
