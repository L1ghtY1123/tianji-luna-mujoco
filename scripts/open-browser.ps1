$ErrorActionPreference = 'Stop'

function Connect-WebTunnel([int]$Port, [string]$Target) {
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        return
    }
    $key = Join-Path $env:USERPROFILE '.ssh\codex_tianji_audit'
    if (-not (Test-Path -LiteralPath $key -PathType Leaf)) {
        throw "SSH key not found: $key"
    }
    $forward = '127.0.0.1:{0}:{1}' -f $Port, $Target
    $sshArguments = @('-N', '-L', $forward, '-i', ('"{0}"' -f $key),
        '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
        '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=8',
        '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
        'guangyu@192.168.3.4')
    $tunnel = Start-Process -FilePath (Get-Command ssh.exe).Source -ArgumentList $sshArguments -WindowStyle Hidden -PassThru
    $deadline = [DateTime]::UtcNow.AddSeconds(15)
    while ([DateTime]::UtcNow -lt $deadline) {
        $tunnel.Refresh()
        if ($tunnel.HasExited) {
            throw "SSH tunnel failed for local port $Port. Check the SSH key and server connection."
        }
        if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
            return
        }
        Start-Sleep -Milliseconds 250
    }
    Stop-Process -InputObject $tunnel
    throw "Timed out opening local port $Port."
}

try {
    # The embedded Base console needs its own local forwarding as well.
    Connect-WebTunnel 8000 '172.29.79.2:8000'
    Connect-WebTunnel 8350 '127.0.0.1:8350'
    $state = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/extensions/jodell.robot-bridge/v1/state' -TimeoutSec 10
    if ($state.ok -ne $true -or $state.runtime_mode -ne 'simulation' -or $state.robot_ip -ne '172.29.79.3') {
        throw 'Local port 8000 is not the configured simulation Base; existing connections were left unchanged.'
    }
    $progress = Invoke-RestMethod -Uri 'http://127.0.0.1:8350/api/run-progress' -TimeoutSec 10
    if ($progress.PSObject.Properties.Name -notcontains 'run' -or $progress.PSObject.Properties.Name -notcontains 'server_time') {
        throw 'Local port 8350 is not the Luna service; existing connections were left unchanged.'
    }
    Start-Process 'http://127.0.0.1:8350'
}
catch {
    Write-Error ("Cannot open Luna. Start Simulation, Base and Luna on the server first. " + $_.Exception.Message) -ErrorAction Continue
    exit 1
}
