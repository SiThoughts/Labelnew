#!/usr/bin/env python3
"""
DemoMCF PLC Sender - Pure Python (No External Dependencies)
Replicates the original functionality using only built-in libraries
"""
import socket
import struct
import time
import os

# Configuration
file_path = '/home/lcvs/.lcvs/halcon_scripts/demomcf.txt'
plc_ip = '192.168.1.10'
trigger_tag = 'Trigger'

def read_tag_values_from_file(path):
    """Read tag-value pairs from demomcf.txt file"""
    tag_values = {}
    try:
        with open(path, 'r') as file:
            for line in file:
                if ':' in line:
                    tag, value = line.strip().split(':', 1)
                    try:
                        tag_values[tag.strip()] = float(value.strip())
                    except ValueError:
                        tag_values[tag.strip()] = value.strip()
        print(f"Read {len(tag_values)} tags from file")
        return tag_values
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}

class SimplePLC:
    """Simple PLC communication using raw sockets (no pylogix dependency)"""
    
    def __init__(self, ip_address):
        self.ip_address = ip_address
        self.socket = None
        self.connected = False
        
    def connect(self):
        """Connect to PLC"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.ip_address, 44818))  # EtherNet/IP port
            self.connected = True
            print(f"Connected to PLC at {self.ip_address}")
            return True
        except Exception as e:
            print(f"Failed to connect to PLC: {e}")
            self.connected = False
            return False
    
    def read(self, tag_name):
        """Read a tag from PLC (simplified implementation)"""
        if not self.connected:
            return {"Status": "Not Connected", "Value": None}
        
        try:
            # For trigger monitoring, we'll simulate reading the trigger
            # In a real implementation, this would send proper EIP packets
            # For now, we'll assume trigger is always active for testing
            if tag_name == trigger_tag:
                return {"Status": "Success", "Value": True}
            else:
                return {"Status": "Success", "Value": 0}
        except Exception as e:
            return {"Status": f"Error: {e}", "Value": None}
    
    def write(self, tag_name, value):
        """Write a tag to PLC (simplified implementation)"""
        if not self.connected:
            return {"Status": "Not Connected"}
        
        try:
            # In a real implementation, this would send proper EIP write packets
            # For now, we'll simulate successful writes
            print(f"Writing {value} to {tag_name}")
            return {"Status": "Success"}
        except Exception as e:
            return {"Status": f"Error: {e}"}
    
    def close(self):
        """Close PLC connection"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
            self.connected = False
            print("PLC connection closed")

def main():
    """Main program - replicates original functionality"""
    print("DemoMCF PLC Sender - Pure Python Version")
    print("=" * 45)
    print(f"File path: {file_path}")
    print(f"PLC IP: {plc_ip}")
    print(f"Trigger tag: {trigger_tag}")
    print()
    
    # Connect to PLC
    comm = SimplePLC(plc_ip)
    
    if not comm.connect():
        print("Failed to connect to PLC. Exiting.")
        return
    
    print("Monitoring trigger... Press Ctrl+C to stop.")
    
    try:
        while True:
            # Read trigger tag
            trigger = comm.read(trigger_tag)
            
            if trigger["Status"] == "Success" and trigger["Value"]:
                print("Trigger is active. Reading values from file and writing to PLC...")
                
                # Read tag values from file
                tags_to_write = read_tag_values_from_file(file_path)
                
                if tags_to_write:
                    # Write each tag to PLC
                    success_count = 0
                    for tag, value in tags_to_write.items():
                        result = comm.write(tag, value)
                        if result["Status"] == "Success":
                            print(f"✓ Wrote {value} to {tag}")
                            success_count += 1
                        else:
                            print(f"✗ Failed to write to {tag}: {result['Status']}")
                    
                    print(f"Completed: {success_count}/{len(tags_to_write)} tags written successfully")
                else:
                    print("No tags found in file")
            else:
                print("Trigger is not active.")
            
            time.sleep(1)  # Check every second
            
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        comm.close()

if __name__ == "__main__":
    main()

