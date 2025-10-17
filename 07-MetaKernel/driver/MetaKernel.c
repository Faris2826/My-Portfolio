#include <ntddk.h>
NTSTATUS DriverEntry(PDRIVER_OBJECT DriverObject, PUNICODE_STRING RegistryPath){
    DbgPrint("[MetaKernel] Driver loaded\n");
    return STATUS_SUCCESS;
}

