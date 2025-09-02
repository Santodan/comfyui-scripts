# Requires Administrator privileges

# Get the name of your Ethernet adapter
$adapterName = "Ethernet 2" # Read-Host "Enter the name of your Ethernet adapter (e.g., 'Ethernet', 'Local Area Connection')"

# Function to set both IPv4 and IPv6 DNS servers
function Set-DnsServers {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Adapter,
        [string[]]$IPv4DnsServers,
        [string[]]$IPv6DnsServers
    )
    try {
        # Set IPv4 to automatic first to clear existing static settings
        $ipv4Command = "netsh interface ip set dns name=`"$Adapter`" dhcp"
        Start-Process -Verb RunAs -FilePath "powershell" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `" $ipv4Command `"" -Wait

        if ($IPv4DnsServers -and $IPv4DnsServers.Count -gt 0) {
            $ipv4Command = "netsh interface ip set dns name=`"$Adapter`" static $($IPv4DnsServers[0])"
            if ($IPv4DnsServers.Count -gt 1 -and $IPv4DnsServers[1]) {
                $ipv4Command += " $($IPv4DnsServers[1])"
            }
            Start-Process -Verb RunAs -FilePath "powershell" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `" $ipv4Command `"" -Wait
        }

        # Set IPv6 to automatic first to clear existing static settings
        $ipv6Command = "netsh interface ipv6 set dns name=`"$Adapter`" dhcp"
        Start-Process -Verb RunAs -FilePath "powershell" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `" $ipv6Command `"" -Wait

        if ($IPv6DnsServers -and $IPv6DnsServers.Count -gt 0) {
            $ipv6Command = "netsh interface ipv6 add dns name=`"$Adapter`" address=$($IPv6DnsServers[0]) index=1"
            if ($IPv6DnsServers.Count -gt 1 -and $IPv6DnsServers[1]) {
                $ipv6Command += "; netsh interface ipv6 add dns name=`"$Adapter`" address=$($IPv6DnsServers[1]) index=2"
            }
            Start-Process -Verb RunAs -FilePath "powershell" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `" $ipv6Command `"" -Wait
        }

        # Add a small delay to ensure the changes are applied before displaying
        Start-Sleep -Seconds 5

        Write-Host "DNS servers (IPv4 and IPv6) for '$Adapter' have been successfully set to:"
        Get-DnsClientServerAddress -InterfaceAlias $Adapter | Format-Table

    } catch {
        Write-Error "An error occurred while trying to set the DNS servers (IPv4 and IPv6) for '$Adapter': $($_.Exception.Message)"
    }
}

# Display options and get user choice
Write-Host "Choose a DNS configuration for '$adapterName':"
Write-Host "1. Automatic (DHCP for both IPv4 and IPv6)"
Write-Host "2. Google DNS (IPv4: 8.8.8.8, 8.8.4.4 | IPv6: 2001:4860:4860::8888, 2001:4860:4860::8844)"
Write-Host "3. Cloudflare DNS (IPv4: 1.1.1.1, 1.0.0.1 | IPv6: 2606:4700:4700::1111, 2606:4700:4700::1001)"

$choice = Read-Host "Enter your choice (1, 2, or 3)"

switch ($choice) {
    "1" {
        Write-Host "Setting DNS to Automatic (DHCP for both IPv4 and IPv6)..."
        # Set DNS to automatic (DHCP)
        netsh interface ipv4 set dns name="Ethernet 2" source=dhcp
        netsh interface ipv6 set dns name="Ethernet 2" source=dhcp
    }
    "2" {
        Write-Host "Setting DNS to Google DNS (IPv4 and IPv6)..."
        Set-DnsClientServerAddress -InterfaceAlias "Ethernet 2" -ServerAddresses ("8.8.8.8", "8.8.4.4")
        Set-DnsClientServerAddress -InterfaceAlias "Ethernet 2" -ServerAddresses ("2001:4860:4860::8888", "2001:4860:4860::8844")
    }
    "3" {
        Write-Host "Setting DNS to Cloudflare DNS (IPv4 and IPv6)..."
        Set-DnsClientServerAddress -InterfaceAlias "Ethernet 2" -ServerAddresses ("1.1.1.1", "1.0.0.1")
        Set-DnsClientServerAddress -InterfaceAlias "Ethernet 2" -ServerAddresses ("2606:4700:4700::1111", "2606:4700:4700::1001")
    }
    default {
        Write-Warning "Invalid choice. No DNS settings were changed."
    }
}

Write-Host "DNS servers (IPv4 and IPv6) for '$adapterName' have been successfully set to:"
Get-DnsClientServerAddress -InterfaceAlias $adapterName | Format-Table
Write-Host "The script has finished running."
Read-Host -Prompt "Press Enter to exit"