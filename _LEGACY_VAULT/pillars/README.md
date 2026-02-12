# NeuralTrader 4-Pillar Architecture
## Professional Isolated-Pillar System

---

## 🏛️ **PILLAR OVERVIEW**

### **📊 Pillar 1: Data Mastery**
- **Script**: `data_mastery.py`
- **Purpose**: Tiingo fetch, cleaning, cache updates
- **Schedule**: Daily 16:45 IST (09:45 EST)
- **Output**: Clean data cache + Pulse email
- **Dependencies**: Tiingo API, YFinance fallback

### **🧠 Pillar 2: The Brain**
- **Script**: `the_brain.py`
- **Purpose**: Training/Retraining AI models
- **Schedule**: Saturday 09:10 IST
- **Output**: Updated models + Pulse email
- **Dependencies**: Clean data, training pipeline

### **⚡ Pillar 3: The Execution**
- **Script**: `the_execution.py`
- **Purpose**: Signal -> Risk -> Trade execution
- **Schedule**: Daily 23:15 IST (15:15 EST)
- **Output**: Portfolio updates + Executive Brief
- **Dependencies**: AI models, risk rules, portfolio

### **📢 Pillar 4: The Voice**
- **Script**: `the_voice.py`
- **Purpose**: Email notifications and logging
- **Schedule**: On-demand (called by other pillars)
- **Output**: Professional emails + daily logs
- **Dependencies**: All pillars for notifications

---

## 🔧 **EXECUTION INSTRUCTIONS**

### **Manual Execution:**
```bash
# Run individual pillars
python pillars/data_mastery.py
python pillars/the_brain.py
python pillars/the_execution.py

# The Voice is called automatically by other pillars
```

### **Task Scheduler Setup:**
```batch
# Windows Task Scheduler tasks:
Task 1: Daily 16:45 IST -> python pillars\data_mastery.py
Task 2: Saturday 09:10 IST -> python pillars\the_brain.py
Task 3: Daily 23:15 IST -> python pillars\the_execution.py
```

### **Pre-Flight Check:**
Every pillar automatically runs `scripts/pre_flight.py` before execution.
If pre-flight fails, the pillar aborts and alerts the Architect.

---

## 📋 **CONFIGURATION**

### **Main Config**: `config/config.yaml`
- All system settings in one place
- Environment-specific configurations
- Broker mode switching (VIRTUAL/LIVE)

### **Secrets**: `config/.env`
- API keys and passwords
- Never commit to version control
- Use `.env.example` as template

---

## 📁 **FILE STRUCTURE**

```
pillars/
├── README.md                 # This file
├── data_mastery.py          # Pillar 1: Data fetching
├── the_brain.py             # Pillar 2: Model training
├── the_execution.py         # Pillar 3: Trading execution
└── the_voice.py             # Pillar 4: Notifications
```

---

## 🚀 **STATUS**

### **✅ COMPLETED:**
- Architecture design
- Configuration system
- Pre-flight integrity checks
- Unified logging system
- Directory structure

### **🔄 IN PROGRESS:**
- Pillar implementation (data_mastery.py)
- Testing and validation
- Task Scheduler integration

### **📋 NEXT STEPS:**
1. Complete all 4 pillar scripts
2. Test end-to-end functionality
3. Set up Task Scheduler
4. Migrate from old architecture
5. Remove redundant files

---

## 🛡️ **SAFETY FEATURES**

### **Pre-Flight Validation:**
- Environment checks
- Configuration validation
- API connectivity tests
- Model file verification
- Directory permissions

### **Fail-Fast Design:**
- Critical failures abort immediately
- Architect notified via email
- Detailed error logging
- System state preserved

### **Accumulative Logging:**
- Single daily log file: `logs/NT_YYYY-MM-DD.log`
- All pillars write to same file
- Complete audit trail
- Email attachments for reports

---

## 📞 **SUPPORT**

### **Troubleshooting:**
1. Check daily log file: `logs/NT_YYYY-MM-DD.log`
2. Run pre-flight manually: `python scripts/pre_flight.py`
3. Verify configuration: `config/config.yaml`
4. Check environment variables: `config/.env`

### **Alert System:**
- Critical failures: Immediate email to Architect
- Pulse emails: After Data Mastery and Brain Training
- Executive Brief: After Execution pillar
- All emails include log file attachments

---

**🏛️ STATUS: 4-PILLAR ARCHITECTURE IMPLEMENTATION IN PROGRESS**
